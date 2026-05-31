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

def get_format_variant_effect_function(model_name: str) -> Any:
    """
    获取Variant Effect数据的格式化函数

    根据模型类型返回对应的格式化函数:
    - "llm": 纯文本模式 (DNA序列转为文本描述)
    - "dna-llm": DNA模式 (使用DNA编码器)

    Args:
        model_name: 模型类型 ("llm" 或 "dna-llm")

    Returns:
        格式化函数
    """
    if model_name.lower() == "llm":
        return format_variant_effect_for_llm
    elif model_name.lower() == "dna-llm":
        return format_variant_effect_for_dna_llm
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


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

def format_variant_effect_for_dna_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    为DNA-LLM格式化Variant Effect数据

    将原始Variant Effect数据转换为对话格式:
    - 用户消息: 包含2个DNA序列(参考+变异) + 问题文本
    - 助手消息: 包含答案 (用于SFT训练)

    对话格式:
        User: [DNA] [DNA] [Question]
        Assistant: Answer: xxx

    数据结构:
        prompt: [
            {"role": "user", "content": [dna, dna, text]},
            {"role": "assistant", "content": [text]}
        ]
        dna_sequences: [reference_sequence, variant_sequence]
        answer: 答案字符串

    Args:
        example: 原始Variant Effect数据示例

    Returns:
        格式化后的数据字典
    """
    return {
        # 用户消息: 混合内容 (2个DNA序列 + 1个文本问题)
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),  # 2个DNA占位符
                    {"type": "text", "text": example["question"].strip()}, # 问题
                ],
            },
            # 助手消息: 模型需要学习的回复
            {
                "role": "assistant",
                "reasoning_content": f"Answer: {example['answer'].strip()}",  # 推理过程/答案
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},  # 最终答案
                ],
            },
        ],
        # DNA序列列表 (对应prompt中的DNA占位符)
        "dna_sequences": [
            example["reference_sequence"],  # 参考序列
            example["variant_sequence"],     # 变异序列
        ],
        "answer": example["answer"].strip(),  # 答案
    }


def format_variant_effect_for_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    为纯LLM格式化Variant Effect数据

    与DNA-LLM模式的区别:
    - 不使用DNA编码器
    - 将DNA序列转为纯文本描述
    - dna_sequences 为空列表

    对话格式:
        User: Reference sequence: xxx\nVariant sequence: xxx\nQuestion: xxx
        Assistant: Answer: xxx

    Args:
        example: 原始Variant Effect数据示例

    Returns:
        格式化后的数据字典
    """
    # 将DNA序列转为文本描述
    question = f"Reference sequence: {example['reference_sequence']}\nVariant sequence: {example['variant_sequence']}\nQuestion: {example['question']}"

    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),  # 保留2个DNA占位符(虽然不用)
                    {"type": "text", "text": question.strip()},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": f"{example['answer'].strip()}",
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            },
        ],
        # LLM模式: DNA序列为空
        "dna_sequences": [
            "",  # 参考序列为空
            "",  # 变异序列为空
        ],
        "answer": example["answer"].strip(),
    }
