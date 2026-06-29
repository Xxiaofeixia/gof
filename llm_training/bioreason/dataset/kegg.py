"""
KEGG 数据集处理模块。
=============================================================================
KEGG 是 Kyoto Encyclopedia of Genes and Genomes 的缩写，本项目用它来做
生物通路推理任务。

数据集格式:
  每条数据包含:
    - reference_sequence: 参考 DNA 序列
    - variant_sequence:   变异 DNA 序列
    - question:           问题文本
    - answer:             答案文本
    - reasoning:          推理过程（有监督微调的目标输出）

两种数据格式化模式:
  1. LLM 模式:      DNA 序列信息被嵌入到文本中（纯文本输入）
  2. DNA-LLM 模式:  DNA 序列单独处理为 embedding 输入（多模态输入）

核心函数:
  - format_kegg_for_*:  将原始数据格式化为模型输入格式
  - qwen_dna_collate_fn: batch 整理函数，构建 labels（只有 assistant 回复参与 loss 计算）
  - dna_collate_fn:      DNA-Only 模型的 batch 整理函数
=============================================================================
"""

import torch
from typing import Any, Dict, List

from functools import partial

from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.dna_modules.nucleotide_module import NucleotideDNAModule


def get_format_kegg_function(model_name: str, is_sft: bool = True) -> Any:
    """
    根据模型类型返回对应的数据格式化函数。

    Args:
        model_name: "llm" 或 "dna-llm"
        is_sft:     是否为监督微调（SFT）模式，SFT模式会在prompt中加入assistant回复
    """
    if model_name.lower() == "llm":
        return partial(format_kegg_for_llm, is_sft=is_sft)
    elif model_name.lower() == "dna-llm":
        return partial(format_kegg_for_dna_llm, is_sft=is_sft)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def _format_kegg(example: Dict[str, Any], model_name: str, is_sft: bool) -> Dict[str, Any]:
    """
    核心格式化函数：将原始 KEGG 样例转为模型的 chat 格式。

    输出格式:
    {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "dna", "text": None},       # DNA序列1占位
                    {"type": "dna", "text": None},       # DNA序列2占位
                    {"type": "text", "text": "question"} # 文本问题
                ]
            },
            {  # 仅在 SFT 模式下存在
                "role": "assistant",
                "reasoning_content": "...",              # 推理过程
                "content": [{"type": "text", "text": "Answer: ..."}]
            }
        ],
        "dna_sequences": ["reference_sequence", "variant_sequence"],
        "answer": "答案"
    }

    两种模式的区别:
    - LLM 模式:    DNA 序列被直接拼接到问题文本中（如 "Reference sequence: ATGC..."）
    - DNA-LLM 模式: DNA 序列单独存在 dna_sequences 字段中，文本只保留问题
    """
    if model_name.lower() not in ['llm', 'dna-llm']:
        raise ValueError(f"Unsupported model name: {model_name}")

    if model_name.lower() == 'llm':
        # 纯文本模式：把DNA序列文本化，嵌入到question里
        question = f"Reference sequence: {example['reference_sequence']}\nVariant sequence: {example['variant_sequence']}\nQuestion: {example['question']}"
        reference_sequence = ""
        variant_sequence = ""
    elif model_name.lower() == 'dna-llm':
        # 多模态模式：DNA序列单独处理
        question = example['question']
        reference_sequence = example['reference_sequence']
        variant_sequence = example['variant_sequence']

    # 构建统一格式
    item = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),  # 两条DNA序列的占位标记
                    {"type": "text", "text": question.strip()},
                ],
            }
        ],
        "dna_sequences": [
            reference_sequence,
            variant_sequence,
        ],
        "answer": example["answer"],
    }

    # SFT 模式：在 prompt 后面加上 assistant 的标准回复（包含推理过程）
    if is_sft:
        item['prompt'].append(
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            }
        )
    return item


def format_kegg_for_dna_llm(example: Dict[str, Any], is_sft: bool) -> Dict[str, Any]:
    """DNA-LLM 模式的格式化"""
    return _format_kegg(example, 'dna-llm', is_sft=is_sft)


def format_kegg_for_llm(example: Dict[str, Any], is_sft: bool) -> Dict[str, Any]:
    """纯 LLM 模式的格式化"""
    return _format_kegg(example, 'llm', is_sft=is_sft)


def _truncate_after_assistant_start(text: str) -> str:
    """
    截断函数：保留从开头到 "<|im_start|>assistant\n" 的部分。
    用于推理时只保留 user 输入，截断 assistant 回复（让模型自己生成）。
    """
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx != -1:
        return text[: idx + len(marker)]
    return text


def qwen_dna_collate_fn(
    examples: List[Dict],
    processor: DLProcessor,
    max_length_text: int,
    max_length_dna: int,
    return_answer_in_batch: bool = False,
    truncate_for_generation: bool = True
) -> Dict:
    """
    DNA-LLM 模型的 batch 整理函数（collate function）。

    核心功能:
    1. 将每个样本的 prompt 用 chat template 渲染为文本
    2. 提取 DNA 序列并分词
    3. 通过 processor 统一处理文本和 DNA
    4. 构建 labels tensor —— 只有 assistant 的回复部分参与 loss 计算
       （user 输入部分的 labels 设为 -100，被忽略）

    Labels 构建原理:
    ┌─────────────────────────────────────────────────────────────────┐
    │  input_ids:  [system] [user question] [assistant] [answer] ...  │
    │  labels:     [-100..] [-100........] [answer tokens] [-100...]  │
    │               ↑ 忽略    ↑ 忽略          ↑ 参与loss计算   ↑ 忽略   │
    └─────────────────────────────────────────────────────────────────┘
    """
    dna_module = NucleotideDNAModule()

    # 保留原始结构化 prompts（用于 reward 函数）
    original_prompts = [example["prompt"] for example in examples]

    # Step 1: 用 chat template 渲染 prompt 为文本
    prompts_text = dna_module.prepare_prompt(processing_class=processor, inputs=examples)
    batch_dna_sequences = [example["dna_sequences"] for example in examples]



    # 调试：只打印前2个样本的完整渲染文本，用于验证数据格式
    if not hasattr(qwen_dna_collate_fn, "_debug_count"):
        qwen_dna_collate_fn._debug_count = 0
    for i, text in enumerate(prompts_text):
        token_len = len(processor.tokenizer.encode(text, add_special_tokens=False))
        if qwen_dna_collate_fn._debug_count < 2:
            print(f"===== Sample {i} raw token length: {token_len} =====")
            print(text)
            print(f"===== Sample {i} end =====")
            qwen_dna_collate_fn._debug_count += 1


    # Step 2: 通过 processor 处理文本和 DNA
    batch = processor(
        text=prompts_text,
        batch_dna_sequences=batch_dna_sequences,
        return_tensors="pt",
        padding=True,
        padding_side="left",          # 左填充
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
    )

    # Step 3: 构建 labels —— 初始全部设为 -100（被忽略）
    labels = torch.full_like(batch["input_ids"], -100)

    # Step 4: 找到 assistant 回复的位置
    assistant_start_marker = "<|im_start|>assistant\n"
    im_end_marker = "<|im_end|>"

    assistant_start_token_ids = processor.tokenizer.encode(
        assistant_start_marker, add_special_tokens=False
    )
    im_end_token_ids = processor.tokenizer.encode(
        im_end_marker, add_special_tokens=False
    )

    assistant_marker_tensor = torch.tensor(
        assistant_start_token_ids, device=batch["input_ids"].device
    )
    im_end_marker_tensor = torch.tensor(
        im_end_token_ids, device=batch["input_ids"].device
    )

    assistant_marker_len = len(assistant_start_token_ids)
    im_end_marker_len = len(im_end_token_ids)

    # Step 5: 对每个样本，找到所有 assistant 回复的区间
    for i in range(batch["input_ids"].shape[0]):
        input_ids = batch["input_ids"][i]
        seq_len = input_ids.size(0)

        assistant_sections = []

        # 找到所有 "<|im_start|>assistant\n" 标记的位置
        start_positions = []
        for pos in range(seq_len - assistant_marker_len + 1):
            if torch.all(
                input_ids[pos : pos + assistant_marker_len] == assistant_marker_tensor
            ):
                start_positions.append(pos + assistant_marker_len)

        # 找到所有 "<|im_end|>" 标记的位置
        end_positions = []
        for pos in range(seq_len - im_end_marker_len + 1):
            if torch.all(
                input_ids[pos : pos + im_end_marker_len] == im_end_marker_tensor
            ):
                end_positions.append(pos)

        # 匹配 start 和 end 标记
        for start_pos in start_positions:
            valid_ends = [pos for pos in end_positions if pos > start_pos]
            if valid_ends:
                end_pos = min(valid_ends) + im_end_marker_len  # 包含 <|im_end|>，让模型学会停止
                if start_pos < end_pos:
                    assistant_sections.append((start_pos, end_pos))
            else:
                assistant_sections.append((start_pos, seq_len))

        # Step 6: 将 assistant 回复区间的 labels 设为对应的 input_ids
        for start_pos, end_pos in assistant_sections:
            if start_pos < end_pos and start_pos < seq_len:
                end_pos = min(end_pos, seq_len)
                labels[i, start_pos:end_pos] = input_ids[start_pos:end_pos]

    # Step 7: 将 padding token 位置的 labels 也设为 -100
    labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    valid_labels = (labels != -100).sum().item()


    # 如果需要返回原始答案
    if return_answer_in_batch:
        batch["answer"] = [example["answer"].strip() for example in examples]

    # 截断：只保留到 assistant 开始的位置（用于推理时空出位置给模型生成）
    prompts_text = [_truncate_after_assistant_start(p) for p in prompts_text]

    # Step 8: 生成模式下截断 prompt，只保留到 "<|im_start|>assistant\n"
    if truncate_for_generation:
        device = batch["input_ids"].device
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id

        composite = "<|im_end|>\n<|im_start|>assistant\n"
        comp_ids = processor.tokenizer.encode(composite, add_special_tokens=False)
        comp_t = torch.tensor(comp_ids, device=device)
        comp_len = len(comp_ids)

        B, L = batch["input_ids"].shape
        keep_lens: List[int] = []

        for i in range(B):
            ids = batch["input_ids"][i]
            keep = L
            for j in range(0, L - comp_len + 1):
                if torch.all(ids[j:j+comp_len] == comp_t):
                    keep = j + comp_len
                    break
            keep_lens.append(keep)

        new_max = max(keep_lens) if keep_lens else 0

        # 重新分配左填充的 tensor
        new_input_ids = torch.full((B, new_max), pad_id, dtype=batch["input_ids"].dtype, device=device)
        new_attention = torch.zeros((B, new_max), dtype=batch["attention_mask"].dtype, device=device)
        new_labels = torch.full((B, new_max), -100, dtype=batch["labels"].dtype, device=device)

        for i, k in enumerate(keep_lens):
            if k == 0:
                continue
            src_ids = batch["input_ids"][i, :k]
            src_attn = batch["attention_mask"][i, :k]
            src_lbls = batch["labels"][i, :k]

            new_input_ids[i, -k:] = src_ids
            new_attention[i, -k:] = src_attn
            new_labels[i, -k:] = src_lbls

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention
        batch["labels"] = new_labels

    batch["prompt"] = prompts_text
    batch["original_prompts"] = original_prompts

    return batch


def dna_collate_fn(
    batch: List[Dict[str, Any]],
    dna_tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    DNA-Only 分类模型的 batch 整理函数。

    与 qwen_dna_collate_fn 不同，这个函数:
    - 不需要文本 LLM
    - 直接对两条 DNA 序列分别分词
    - 返回 ref_ids, alt_ids 和分类标签
    """
    ref_sequences = [item["reference_sequence"] for item in batch]
    alt_sequences = [item["variant_sequence"] for item in batch]

    # 分别对参考序列和变异序列做分词
    tokenized_ref = dna_tokenizer(
        ref_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tokenized_alt = dna_tokenizer(
        alt_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # 标签：将答案文本转为分类ID
    labels = []
    for item in batch:
        label = label2id[item["answer"]]
        labels.append(label)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    tokenized_batch = {
        "ref_ids": tokenized_ref.input_ids,
        "ref_attention_mask": tokenized_ref.attention_mask,
        "alt_ids": tokenized_alt.input_ids,
        "alt_attention_mask": tokenized_alt.attention_mask,
        "labels": labels_tensor,
    }

    return tokenized_batch
