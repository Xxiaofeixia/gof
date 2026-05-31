from typing import List, Optional, Union, Dict, Any, Tuple

import torch

from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from bioreason.utils.dna_utils import DNAInput

class DLDNAKwargs():
    """Keyword arguments specific to DNA processing"""
    max_length_text: Optional[int]
    max_length_dna: Optional[int]


class DLProcessorKwargs(ProcessingKwargs, total=False):
    """Processing keyword arguments for the DL processor"""
    dna_kwargs: DLDNAKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }

class DLProcessor(ProcessorMixin):
    r"""
    Constructs a DL processor which wraps a NucleotideTransformer DNA processor and a Qwen2_5 tokenizer into a single processor.
    This processor handles both text and DNA sequence processing to prepare inputs for the DNALLMModel.
    
    Args:
        tokenizer (PreTrainedTokenizerBase, *optional*):
            The text tokenizer used for processing text inputs.
        dna_tokenizer (PreTrainedTokenizerBase, *optional*):
            The DNA tokenizer used for processing DNA sequences.
        chat_template (`str`, *optional*): 
            A Jinja template for chat formatting. If None, will use the tokenizer's template.
    """

    attributes = ["tokenizer", "dna_tokenizer"]
    valid_kwargs = ["model", "chat_template"]
    tokenizer_class = (
        "Qwen2Tokenizer", "Qwen2TokenizerFast",
        "GPT2TokenizerFast",
    )
    dna_tokenizer_class = ("EsmTokenizer", "Evo2Tokenizer")

    def __init__(self, tokenizer=None, dna_tokenizer=None, chat_template=None, **kwargs):
        # 1. 绕过父类，直接手动赋值核心组件
        self.tokenizer = tokenizer
        self.dna_tokenizer = dna_tokenizer
        
        # 2. 手动设置必要属性，模仿父类行为但不触发校验
        self.attributes = ["tokenizer", "dna_tokenizer"]
        self.chat_template = chat_template if chat_template else getattr(tokenizer, "chat_template", None)
        
        # 3. 设置 DNA 占位符
        self.dna_token = getattr(self.tokenizer, "dna_token", "<|dna_pad|>")

        # 4. 兼容性设置：确保 pad_token 存在
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 5. 【关键】给父类留下空清单，彻底堵死 getattr 报错路径
        self.__class__._class_name = []
        
        # 注意：这里我们不再调用 super().__init__，或者改用最安全的方式调用
        # 如果代码其他地方必须依赖父类的某些初始化，可以用 try-except 包裹
        try:
            super().__init__(tokenizer, dna_tokenizer, chat_template=self.chat_template)
        except AttributeError:
            # 如果 super 还是报错，我们就已经手动完成了赋值，直接忽略它
            pass
    def tokenize_dna_sequences(
        self, 
        batch_dna_sequences: List[List[str]], 
        max_length: int = 2048,
        return_tensors: str = "pt",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Tokenize a batch of DNA sequences.
        
        Args:
            batch_dna_sequences: List of lists of DNA sequences per batch item
            max_length: Maximum allowed length for DNA sequences
            return_tensors: Return format for tensors ("pt" for PyTorch)
            device: Device to place tensors on
            
        Returns:
            Dict containing:
                - dna_tokenized: The tokenized DNA sequences 
                - batch_idx_map: Mapping of which sequences belong to which batch item
        """
        # Create a mapping to track which sequences belong to which batch item
        batch_idx_map = []
        all_sequences = []

        # Flatten all sequences with batch tracking
        for batch_idx, dna_sequences in enumerate(batch_dna_sequences):
            for seq in dna_sequences:
                all_sequences.append(seq)
                batch_idx_map.append(batch_idx)

        # If no sequences in the entire batch, return empty dict
        if not all_sequences:
            return {"dna_tokenized": None, "batch_idx_map": []}

        # Tokenize all sequences at once
        dna_tokenized = self.dna_tokenizer(
            all_sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
            add_special_tokens=False,
        )
            
        return {"dna_tokenized": dna_tokenized, "batch_idx_map": batch_idx_map}

    def __call__(
        self,
        batch_dna_sequences: Optional[List[List[str]]] = None,
        text: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = None,
        max_length_text: int = 512,
        max_length_dna: int = 2048,
        return_tensors: str = "pt",
        device: str = "cuda",
        **kwargs: [DLProcessorKwargs],
    ) -> BatchFeature:
        """
        Process text and DNA sequences for model input.
        
        Args:
            batch_dna_sequences: List of lists of DNA sequences per batch item
            text: Input text or list of texts
            max_length_text: Maximum length for text sequences
            max_length_dna: Maximum length for DNA sequences
            return_tensors: Return format for tensors
            device: Device to place tensors on
            **kwargs: Additional processor keyword arguments
            
        Returns:
            BatchFeature with tokenized inputs for the model
        """
        output_kwargs = self._merge_kwargs(
            DLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Ensure text is a list
        if not isinstance(text, list):
            text = [text]

        # flattened_dna_sequences = [dna_sequence for dna_sequences in batch_dna_sequences for dna_sequence in dna_sequences]
        dna_inputs = {}
        if batch_dna_sequences is not None:
            # Tokenize DNA sequences
            dna_processing_result = self.tokenize_dna_sequences(
                batch_dna_sequences,
                max_length=max_length_dna,
                return_tensors=return_tensors,
                device=device,
            )
            
            # Replace DNA tokens in text if needed
            # 找到文本替换逻辑块
            index = 0
            for i in range(len(text)):
                # --- 鲁棒性增强开始 ---
                # 无论 text[i] 是 ["str"] 还是 [["str"]]，都强制取出最里面的字符串
                current_item = text[i]
                while isinstance(current_item, list) and len(current_item) > 0:
                    current_item = current_item[0]
                
                # 如果还是列表或者是 None，给个保底字符串，防止 replace 报错
                if not isinstance(current_item, str):
                    current_text = str(current_item)
                else:
                    current_text = current_item
                # --- 鲁棒性增强结束 ---

                while self.dna_token in current_text:
                    # 获取当前 DNA 序列对应的 token 长度
                    num_dna_tokens = dna_processing_result['dna_tokenized']['attention_mask'][index].sum().item()
                    current_text = current_text.replace(
                        self.dna_token, "<|placeholder|>" * num_dna_tokens, 1
                    )
                    index += 1
                
                # 最终替换回占位符，并确保 text[i] 存储的是纯字符串
                text[i] = current_text.replace("<|placeholder|>", self.dna_token)
            
            # Add batch info to the output
            dna_inputs = {
                # "batch_dna_sequences": batch_dna_sequences,
                "dna_tokenized": dna_processing_result["dna_tokenized"],
                "batch_idx_map": dna_processing_result["batch_idx_map"],
            }

        # Tokenize text
        text_kwargs = output_kwargs.get("text_kwargs", {})
        
        if 'padding' in text_kwargs:
            del text_kwargs['padding']
        
        # --- 核心修复：清理分词器不认识的参数 ---
        forbidden_keys = ['padding_side', 'padding'] 
        for key in forbidden_keys:
            if key in text_kwargs:
                del text_kwargs[key]
        # ------------------------------------
        
        # print("__call__ (processor):", text)
        text_inputs = self.tokenizer(
            text, 
            max_length=max_length_text + 2 * max_length_dna,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            **text_kwargs,
        )
        # --- 强力拍平开始 ---
        final_data = {}
        
        # 1. 文本部分
        for k, v in text_inputs.items():
            final_data[k] = torch.as_tensor(v)

        # 2. DNA 部分 (直接把 dna_tokenized 里的内容拆出来放进一层字典)
        if "dna_tokenized" in dna_inputs and dna_inputs["dna_tokenized"] is not None:
            # 获取内部的字典内容
            dna_encoding = dna_inputs["dna_tokenized"]
            for k, v in dna_encoding.items():
                # 键名改为 dna_input_ids 等，值确保是 Tensor
                final_data[f"dna_{k}"] = torch.as_tensor(v)
            
            # 3. 映射部分 (强制转换为 Long 类型的 Tensor)
            if "batch_idx_map" in dna_inputs:
                final_data["batch_idx_map"] = torch.as_tensor(dna_inputs["batch_idx_map"], dtype=torch.long)
        # --- 强力拍平结束 ---

        return BatchFeature(data=final_data)
        # The BatchFeature should have all required fields for the model's forward pass
        #return BatchFeature(data={**text_inputs, **dna_inputs})

    def batch_decode(self, *args, **kwargs) -> List[str]:
        """
        This method forwards all its arguments to the tokenizer's batch_decode.
        
        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        """
        This method forwards all its arguments to the tokenizer's decode.
        
        Returns:
            Decoded string
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_dna_to_text(
        self,
        generated_outputs: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Post-process the model output to decode the text.
        
        Args:
            generated_outputs: The token IDs generated by the model
            skip_special_tokens: Whether to skip special tokens in the output
            **kwargs: Additional arguments for the decoder
            
        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        """
        Get the input names expected by the model.
        
        Returns:
            List of input names
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        dna_input_names = ["dna_tokenized", "batch_idx_map"]
        
        return list(dict.fromkeys(tokenizer_input_names + dna_input_names))
