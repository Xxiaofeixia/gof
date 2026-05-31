"""
DLProcessor —— 文本 + DNA 双处理器。
=============================================================================
这是连接数据和模型的桥梁。它同时处理:
  1. 文本: 用 text_tokenizer (Qwen3 tokenizer) 分词
  2. DNA序列: 用 dna_tokenizer (NT/Evo2 tokenizer) 分词

关键流程 (__call__):
  输入: text=["DNA Sequence 1: <|dna_pad|> ..."], batch_dna_sequences=[["ATCG...", "GCTA..."]]
  ──▶ Step 1: 用 dna_tokenizer 分词 DNA 序列
  ──▶ Step 2: 将文本中的 <|dna_pad|> 占位符替换为实际数量个占位符
             (因为每个DNA序列分词后的token数量不同)
  ──▶ Step 3: 用 text_tokenizer 分词文本
  ──▶ 输出: BatchFeature 包含 input_ids, attention_mask, dna_tokenized, batch_idx_map

模型模式兼容:
  - DNA-LLM 模式: dna_tokenizer 正常使用，<|dna_pad|> 被替换
  - LLM 模式: dna_tokenizer=None，<|dna_pad|> 被直接删除（纯文本模式）
=============================================================================
"""

from typing import List, Optional, Union, Dict, Any, Tuple

import torch

from transformers.processing_utils import (
    CommonKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from bioreason.utils.dna_utils import DNAInput


class DLDNAKwargs(CommonKwargs):
    """DNA 处理的关键字参数"""
    max_length_text: Optional[int]
    max_length_dna: Optional[int]


class DLProcessorKwargs(ProcessingKwargs, total=False):
    """处理器关键字参数"""
    dna_kwargs: DLDNAKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class DLProcessor(ProcessorMixin):
    """
    双模态处理器：同时处理文本和 DNA 序列。

    Args:
        tokenizer:     文本分词器（Qwen3 tokenizer）
        dna_tokenizer: DNA 分词器（NT tokenizer 或 Evo2 tokenizer）
        chat_template: 对话模板（Jinja 格式）
    """

    attributes = ["tokenizer", "dna_tokenizer"]
    valid_kwargs = ["model", "chat_template"]
    tokenizer_class = (
        "Qwen2Tokenizer", "Qwen2TokenizerFast",
        "GPT2TokenizerFast",
    )
    dna_tokenizer_class = ("EsmTokenizer", "Evo2Tokenizer")

    def __init__(
        self, tokenizer=None, dna_tokenizer=None, chat_template=None, **kwargs
    ):
        self.tokenizer = tokenizer
        self.dna_tokenizer = dna_tokenizer

        # DNA 占位符 token：文本中的 <|dna_pad|> 会被替换为 DNA embedding
        self.dna_token = (
            "<|dna_pad|>"
            if not hasattr(self.tokenizer, "dna_token")
            else self.tokenizer.dna_token
        )

        if chat_template is None and hasattr(self.tokenizer, "chat_template"):
            chat_template = self.tokenizer.chat_template

        # 兼容 dna_tokenizer=None 的情况（纯LLM模式）
        if dna_tokenizer is not None:
            super().__init__(tokenizer, dna_tokenizer, chat_template=chat_template)
        else:
            self.chat_template = chat_template

        # GRPO trainer 可能需要这个属性
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_dna_sequences(
        self,
        batch_dna_sequences: List[List[str]],
        max_length: int = 2048,
        return_tensors: str = "pt",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        对一个 batch 中所有的 DNA 序列进行分词。

        输入格式: batch_dna_sequences = [
            ["ATCG...", "GCTA..."],  # 样本0的两条DNA序列（reference + variant）
            ["CCGT...", "TAGC..."],  # 样本1的两条DNA序列
        ]

        输出:
            dna_tokenized: tokenized 后的 DNA 序列（包含 input_ids, attention_mask）
            batch_idx_map: 每条DNA序列属于哪个 batch item 的映射 [0, 0, 1, 1, ...]
        """
        batch_idx_map = []
        all_sequences = []

        # 展开所有DNA序列，记录每条属于哪个batch item
        for batch_idx, dna_sequences in enumerate(batch_dna_sequences):
            for seq in dna_sequences:
                all_sequences.append(seq)
                batch_idx_map.append(batch_idx)

        if not all_sequences:
            return {"dna_tokenized": None, "batch_idx_map": []}

        # 纯LLM模式：dna_tokenizer=None，直接返回空
        if self.dna_tokenizer is None:
            return {"dna_tokenized": None, "batch_idx_map": []}

        # DNA-LLM模式：用 dna_tokenizer 分词所有DNA序列
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
        **kwargs: Unpack[DLProcessorKwargs],
    ) -> BatchFeature:
        """
        处理一个 batch 的文本和 DNA 序列，生成模型输入。

        核心逻辑:
        1. 分词 DNA 序列
        2. 将文本中的 <|dna_pad|> 替换为正确数量的占位符
           （因为每个DNA序列分词后产生的 token 数量不同）
        3. 分词文本
        4. 返回合并后的 BatchFeature
        """
        output_kwargs = self._merge_kwargs(
            DLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, list):
            text = [text]

        dna_inputs = {}
        if batch_dna_sequences is not None:
            # Step 1: 分词 DNA
            dna_processing_result = self.tokenize_dna_sequences(
                batch_dna_sequences,
                max_length=max_length_dna,
                return_tensors=return_tensors,
                device=device,
            )

            if dna_processing_result['dna_tokenized'] is not None:
                # Step 2 (DNA-LLM模式): 替换占位符
                # 关键：每个 <|dna_pad|> 需要替换成 num_dna_tokens 个占位符
                # 因为每个DNA序列分词后的长度不同
                index = 0
                for i in range(len(text)):
                    while self.dna_token in text[i]:
                        # 获取当前DNA序列分词后的实际长度
                        num_dna_tokens = int(dna_processing_result['dna_tokenized']['attention_mask'][index].sum().item())

                        # 先用临时占位符替换，避免无限循环
                        text[i] = text[i].replace(
                            self.dna_token, "<|placeholder|>" * num_dna_tokens, 1
                        )
                        index += 1
                    # 最终替换回 dna_token
                    text[i] = text[i].replace("<|placeholder|>", self.dna_token)
            else:
                # Step 2 (纯LLM模式): 直接删除 <|dna_pad|>
                for i in range(len(text)):
                    text[i] = text[i].replace(self.dna_token, "")

            dna_inputs = {
                "dna_tokenized": dna_processing_result["dna_tokenized"],
                "batch_idx_map": dna_processing_result["batch_idx_map"],
            }

        # Step 3: 分词文本
        text_kwargs = output_kwargs.get("text_kwargs", {})

        if 'padding' in text_kwargs:
            del text_kwargs['padding']

        # 文本最大长度 = max_length_text + 2 * max_length_dna
        # （为DNA序列的token预留空间）
        text_inputs = self.tokenizer(
            text,
            max_length=max_length_text + 2 * max_length_dna,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            **text_kwargs,
        )

        # Step 4: 合并返回
        return BatchFeature(data={**text_inputs, **dna_inputs})

    def batch_decode(self, *args, **kwargs) -> List[str]:
        """批量解码（代理给 tokenizer）"""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        """解码单个序列（代理给 tokenizer）"""
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_dna_to_text(
        self,
        generated_outputs: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """后处理：将生成的 token IDs 解码为文本"""
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        """模型期望的所有输入名称"""
        tokenizer_input_names = self.tokenizer.model_input_names
        dna_input_names = ["dna_tokenized", "batch_idx_map"]

        return list(dict.fromkeys(tokenizer_input_names + dna_input_names))
