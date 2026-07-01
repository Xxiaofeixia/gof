"""
Variant Effect 数据处理模块
==========================

功能:
    - Variant Effect (变异效应) 预测任务的数据处理
    - 支持两种数据集: coding 和 non-snv
    - 数据清洗和格式化

数据来源:
    - HuggingFace: wanglab/variant_effect_coding (编码区变异)
    - HuggingFace: wanglab/variant_effect_non_snv (非单核苷酸变异，如插入缺失)

任务说明:
    - 输入: DNA参考序列 + 变异序列 + 问题
    - 输出: 变异效应分类 (如 "pathogenic"/"benign" 或具体效应描述)

原始数据字段:
    - reference_sequence: 参考DNA序列
    - variant_sequence: 变异后DNA序列
    - question: 问题文本
    - answer: 变异效应答案

⚠️ 注意: 本模块存在数据泄露问题 (详见 train_dna_qwen.py 注释)
"""

import json
import os
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Tuple


# ==================== 格式化函数获取 ====================

def get_format_variant_effect_function(model_name: str, is_sft: bool = True) -> Any:
    """
    获取Variant Effect数据的格式化函数

    根据模型类型返回对应的格式化函数:
    - "llm": 纯文本模式 (DNA序列转为文本描述)
    - "dna-llm": DNA模式 (使用DNA编码器)

    Args:
        model_name: 模型类型 ("llm" 或 "dna-llm")
        is_sft:     是否为监督微调模式; False 时跳过 assistant 回复 (GRPO 用)

    Returns:
        格式化函数
    """
    if model_name.lower() == "llm":
        fn = format_variant_effect_for_llm
    elif model_name.lower() == "dna-llm":
        fn = format_variant_effect_for_dna_llm
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    from functools import partial
    return partial(fn, is_sft=is_sft)


# ==================== 数据清洗函数 ====================

def clean_variant_effect_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    清洗 Variant Effect 示例的答案字段

    处理逻辑:
    1. 按分号分割答案，取第一部分 (可能有多个答案用分号分隔)
    2. 去除首尾空格
    3. 转为小写

    示例:
        "Pathogenic;benign" -> "pathogenic"

    Args:
        example: 数据示例

    Returns:
        清洗后的示例
    """
    example['answer'] = example['answer'].split(";")[0].strip().lower()
    return example


def get_reasoning_text(example: Dict[str, Any]) -> str:
    """优先使用第 10 步生成的扁平化 SFT 推理链，兼容旧版 reasoning 列。"""
    for key in ("reasoning_sft", "reasoning"):
        value = example.get(key, "")
        if value and str(value).strip():
            return str(value).strip()
    return ""


def clean_variant_effect_non_snv_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    清洗 Non-SNV Variant Effect 示例的答案字段

    处理逻辑:
    1. 移除方括号 []
    2. 移除单引号 '
    3. 将下划线转为空格
    4. 去除首尾空格

    示例:
        "['insertion']" -> "insertion"
        "['deletion']" -> "deletion"
        "pathogenic_variant" -> "pathogenic variant"

    Args:
        example: 数据示例

    Returns:
        清洗后的示例
    """
    example['answer'] = example['answer'].replace("[", "").replace("]", "").replace("'", "").replace("_", " ").strip()
    return example


# ==================== 数据格式化函数 ====================

def format_variant_effect_for_dna_llm(example: Dict[str, Any], is_sft: bool = True) -> Dict[str, Any]:
    """
    为DNA-LLM格式化Variant Effect数据

    将原始Variant Effect数据转换为对话格式:
    - 用户消息: 包含2个DNA序列(参考+变异) + 问题文本
    - 助手消息(is_sft=True时): 包含答案 (用于SFT训练)

    对话格式:
        User: [DNA] [DNA] [Question]
        Assistant: Answer: xxx

    数据结构:
        prompt: [
            {"role": "user", "content": [dna, dna, text]},
            {"role": "assistant", "content": [text]}   # 仅 is_sft=True
        ]
        dna_sequences: [reference_sequence, variant_sequence]
        answer: 答案字符串

    Args:
        example: 原始Variant Effect数据示例
        is_sft:  是否为SFT模式; False=GRPO模式(不含assistant回复)

    Returns:
        格式化后的数据字典
    """
    item = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": example["question"].strip()},
                ],
            },
        ],
        "dna_sequences": [
            example["reference_sequence"],
            example["variant_sequence"],
        ],
        "answer": example["answer"].strip(),
    }
    if is_sft:
        reasoning = get_reasoning_text(example)
        if reasoning:
            item["prompt"].append({
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            })
        else:
            # 无推理链时用答案文本作为 fallback，避免 chat template 渲染空 <think>
            fallback = f"Answer: {example['answer'].strip()}"
            item["prompt"].append({
                "role": "assistant",
                "reasoning_content": fallback,
                "content": [
                    {"type": "text", "text": fallback},
                ],
            })
    return item


def format_variant_effect_for_llm(example: Dict[str, Any], is_sft: bool = True) -> Dict[str, Any]:
    """
    为纯LLM格式化Variant Effect数据

    与DNA-LLM模式的区别:
    - 不使用DNA编码器
    - 将DNA序列转为纯文本描述
    - dna_sequences 为空列表

    Args:
        example: 原始Variant Effect数据示例
        is_sft:  是否为SFT模式; False=GRPO模式(不含assistant回复)

    Returns:
        格式化后的数据字典
    """
    question = f"Reference sequence: {example['reference_sequence']}\nVariant sequence: {example['variant_sequence']}\nQuestion: {example['question']}"

    item = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": question.strip()},
                ],
            },
        ],
        "dna_sequences": ["", ""],
        "answer": example["answer"].strip(),
    }
    if is_sft:
        reasoning = get_reasoning_text(example)
        if reasoning:
            item["prompt"].append({
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            })
        else:
            fallback = f"Answer: {example['answer'].strip()}"
            item["prompt"].append({
                "role": "assistant",
                "reasoning_content": fallback,
                "content": [
                    {"type": "text", "text": fallback},
                ],
            })
    return item
