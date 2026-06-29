import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from evo2 import Evo2
from vortex.model.model import StripedHyena

from typing import Optional, List, Dict, Any, Union, Tuple

from bioreason.utils.dna_utils import DNAInput
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE
from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer

register_evo2_tokenizer()

def get_target_modules(model):
    # Apply LoRA to all linear layers in the text model
    target_modules = []

    # Get all unique linear layer names
    seen_names = set()
    for name, module in model.text_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            target_name = names[-1]  # Use the last part of the name

            # Skip output head but include all other linear layers
            if target_name != "lm_head" and target_name not in seen_names:
                target_modules.append(target_name)
                seen_names.add(target_name)

    # Add attention-specific layers
    attention_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]
    for pattern in attention_patterns:
        if pattern not in seen_names:
            target_modules.append(pattern)

    # Return all unique layer names to apply LoRA to all layers
    return list(target_modules)




class DNALLMModel(nn.Module):
    """
    A combined model that processes both DNA sequences and text inputs.

    The model uses a DNA encoder (like NucleotideTransformer) to extract features from DNA sequences
    and a text model (LLM) to process text inputs and generate responses. The DNA features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        dna_model_name: str,
        cache_dir: Optional[str] = None,
        max_length_dna: int = 2048,
        max_length_text: int = 512,
        text_model_finetune: bool = True,
        dna_model_finetune: bool = True,
        dna_is_evo2: bool = False,
        dna_embedding_layer: str = None,
        device: str = "cuda",
    ):
        """
        Initialize the DNALLMModel.
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_model_finetune = text_model_finetune
        self.dna_model_finetune = dna_model_finetune
        self.max_length_dna = max_length_dna
        self.max_length_text = max_length_text
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer

        # Load the text model and tokenizer
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map=device
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            trust_remote_code=True
        )
        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = CHAT_TEMPLATE
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.dna_token_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")

        # Load the DNA model and tokenizer
        if not self.dna_is_evo2:
            self.dna_model = AutoModelForMaskedLM.from_pretrained(
                dna_model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
            self.dna_config = self.dna_model.config

        else:

            from vortex.model.model import StripedHyena
            from vortex.model.utils import dotdict
            import pkgutil
            import yaml

            # �ֶ����� config
            config_path = "configs/evo2-1b-8k.yml"

            config = yaml.safe_load(pkgutil.get_data("evo2", config_path))
            config = dotdict(config)
            config.use_fp8_input_projections = False
            config.use_flash_attn = False
            config.use_flashfft = False
            config.use_flash_rmsnorm = False

            # �ֶ�����ģ�Ͳ�����Ȩ��
            local_path = "/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/evo2_1b_base/evo2_1b_base.pt"
            dna_backbone = StripedHyena(config)
            state_dict = torch.load(local_path, map_location="cpu")
            # ���ݲ�ͬ��checkpoint��ʽ
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            dna_backbone.load_state_dict(state_dict, strict=False)

            # �ҵ� dna_model �ϱ��ֽӿ�һ��
            self.dna_model = Evo2.__new__(Evo2)
            self.dna_model.model = dna_backbone
            self.dna_model.tokenizer = __import__("vortex.model.tokenizer", fromlist=["CharLevelTokenizer"]).CharLevelTokenizer(512)

            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_config = config
            self.dna_embedding_layer = self.dna_embedding_layer

        # Get model dimensions
        self.text_hidden_size = self.text_config.hidden_size
        # ���޸��㡿������ Evo2 �� hidden_size ��ȡ��ʽ
        self.dna_hidden_size = getattr(self.dna_config, 'hidden_size', getattr(self.dna_config, 'd_model', 2048))

        # Create projection layer to map DNA embeddings to text model's embedding space
        self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size)

        # Create processor for handling inputs
        self.processor = DLProcessor(tokenizer=self.text_tokenizer, dna_tokenizer=self.dna_tokenizer)


    def process_dna_embeddings(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:

        if self.dna_is_evo2:
            dna_device = next(self.dna_model.model.parameters()).device
            dna_tokenized = {
                k: v.to(dna_device) if isinstance(v, torch.Tensor) else v
                for k, v in dna_tokenized.items()
            }

            hidden_states_list = []
            for seq_idx in range(len(dna_tokenized["input_ids"])):
                input_ids = dna_tokenized["input_ids"][seq_idx:seq_idx+1]

                captured = {}
                def hook_fn(m, i, o):
                    captured['out'] = o[0] if isinstance(o, tuple) else o

                last_block = self.dna_model.model.blocks[-1]
                handle = last_block.register_forward_hook(hook_fn)


                with torch.no_grad(), torch.autocast(device_type=dna_device.type, dtype=torch.bfloat16):
                    self.dna_model.model.forward(input_ids)

                handle.remove()
                seq_embeddings = captured['out'].squeeze(0).detach()
                seq_embeddings = seq_embeddings.to(dtype=self.dna_projection.weight.dtype)
                hidden_states_list.append(seq_embeddings)

            hidden_states = torch.stack(hidden_states_list)

        else:
            dna_device = next(self.dna_model.parameters()).device
            dna_tokenized = {
                k: v.to(dna_device) if isinstance(v, torch.Tensor) else v
                for k, v in dna_tokenized.items()
            }


            with torch.no_grad(), torch.autocast(device_type=dna_device.type, dtype=torch.bfloat16):
                outputs = self.dna_model(
                    input_ids=dna_tokenized["input_ids"],
                    attention_mask=dna_tokenized["attention_mask"],
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

        # Project all embeddings at once
        hidden_states = hidden_states.to(
            device=self.dna_projection.weight.device,
            dtype=self.dna_projection.weight.dtype
        )
        projected_states = self.dna_projection(hidden_states)

        # Group embeddings by batch item
        result = [[] for _ in range(2 * batch_size)]
        for seq_idx, batch_idx in enumerate(batch_idx_map):
            if self.dna_is_evo2:
                seq_embedding = projected_states[seq_idx]
            else:
                valid_length = dna_tokenized["attention_mask"][seq_idx].sum().item()
                seq_embedding = projected_states[seq_idx, :valid_length]
            result[batch_idx].append(seq_embedding)

        for i in range(2 * batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                result[i] = torch.zeros(
                    (0, self.text_hidden_size),
                    device=self.dna_projection.weight.device,
                    dtype=self.dna_projection.weight.dtype
                )

        return result

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text based on DNA and text inputs.
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("Either 'inputs' or 'input_ids'/'attention_mask' must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            # Process dna sequences to get embeddings
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)

            mask = input_ids == self.dna_token_id
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)

            # Ensure DNA embeddings have the same dtype and device as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device)
            text_inputs_embeds[mask] = dna_embeds_flat

        # Handle labels if provided (for training)
        if labels is not None:
            # TODO: Implement this
            pass

        # Forward pass through the text model (loss is computed if labels is provided)
        outputs = self.text_model(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on DNA and text inputs.
        """
        text_inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map
        )

        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)
        im_end_id = self.text_tokenizer.convert_tokens_to_ids("<|im_end|>")
        # 移除 TRL 特有参数，不能透传�?text_model.generate
        generation_kwargs.pop("eos_token_id", None)
        generation_kwargs.pop("original_prompts", None)

        with torch.no_grad():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                repetition_penalty=1.0,       # 不惩罚已出现的token, 避免压制 <|im_end|>
                eos_token_id=[self.text_tokenizer.eos_token_id, im_end_id],
                pad_token_id=self.text_tokenizer.eos_token_id,
                **generation_kwargs,
            )

        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        use_reentrant = (
            gradient_checkpointing_kwargs["use_reentrant"]
        )

        print("use_reentrant:", use_reentrant)

        if use_reentrant:
            self.text_model.enable_input_require_grads()

        print("gradient_checkpointing_enable for model:", self.text_model.is_gradient_checkpointing)

    @property
    def is_gradient_checkpointing(self):
        return self.text_model.is_gradient_checkpointing

    def gradient_checkpointing_disable(self):
        self.text_model.gradient_checkpointing_disable()

    def train(self, mode: bool = True):
        nn.Module.train(self, False)
        self.text_model.train(mode)
        self.dna_projection.train(mode)

        if hasattr(self, "lm_head"):
            self.lm_head.train(mode)
        self.training = self.text_model.training
        return self

    def get_prompt_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None
    ):
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            dna_tokenized = {k: v.to(self.device) for k, v in dna_tokenized.items()}
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)

            mask = input_ids == self.dna_token_id
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)

            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device)
            text_inputs_embeds[mask] = dna_embeds_flat

        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        return text_inputs_embeds, attention_mask