"""
dna_llm_text_only.py
纯LLM模型，去除所有DNA相关组件，用于消融实验对比。
基于 dna_llm.py 修改，不影响原始文件。
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Optional, List, Dict, Any, Union

from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE


def get_target_modules(model):
    """Apply LoRA to all linear layers in the text model."""
    target_modules = []
    seen_names = set()
    for name, module in model.text_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_name = name.split(".")[-1]
            if target_name != "lm_head" and target_name not in seen_names:
                target_modules.append(target_name)
                seen_names.add(target_name)

    attention_patterns = ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"]
    for pattern in attention_patterns:
        if pattern not in seen_names:
            target_modules.append(pattern)

    return list(target_modules)


class LLMOnlyModel(nn.Module):
    """
    纯LLM模型，不包含DNA编码器和projection层。
    用于消融实验，与 DNALLMModel 做对比。
    """

    def __init__(
        self,
        text_model_name: str,
        cache_dir: Optional[str] = None,
        max_length_text: int = 512,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length_text = max_length_text

        # 只加载文本模型和tokenizer
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map=device,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            trust_remote_code=True,
        )
        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = CHAT_TEMPLATE
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # 保持与 DNALLMModel 一致，添加相同的special tokens
        # （数据集格式化函数可能会用到这些token）
        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

        # 无DNA组件
        self.dna_model = None
        self.dna_tokenizer = None
        self.dna_projection = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized=None,       # 接收但忽略，保持接口兼容
        batch_idx_map=None,       # 接收但忽略，保持接口兼容
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        # 直接用text embedding，忽略DNA输入
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

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
        dna_tokenized=None,       # 接收但忽略，保持接口兼容
        batch_idx_map=None,       # 接收但忽略，保持接口兼容
        **generation_kwargs,
    ):
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        with torch.no_grad():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        if gradient_checkpointing_kwargs.get("use_reentrant", False):
            self.text_model.enable_input_require_grads()

    def train(self, mode: bool = True):
        nn.Module.train(self, False)
        self.text_model.train(mode)
        self.training = self.text_model.training
        return self
