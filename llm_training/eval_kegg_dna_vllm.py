#!/usr/bin/env python
"""
DNA-vLLM 模型在 KEGG 测试集上的评估脚本。
=============================================================================
功能：加载训练好的 DNA-LLM checkpoint，用 vLLM 进行高效推理，
      在 KEGG 测试集上评估生物通路推理任务的性能。

评估流程：
  1. 加载 KEGG val + test 数据集
  2. 初始化 DNALLMModel（vLLM 版本），加载 checkpoint 权重
  3. 逐条推理，提取 "Answer:" 后的内容作为预测答案
  4. 计算 accuracy、precision、recall、F1 等指标
  5. 保存结果到 JSON/CSV 文件

与训练时的区别：
  - 使用 dna_vllm.py 中的 DNALLMModel（vLLM 推理版本）
  - dna_vllm.py 的模型内置了 vLLM 的 LLM 引擎，推理速度极快
  - 不需要训练循环，只需要前向生成
=============================================================================
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bioreason.models.dna_vllm import DNALLMModel
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import format_kegg_for_dna_llm
from trl.data_utils import maybe_apply_chat_template
from datasets import load_dataset, concatenate_datasets


def load_kegg_test_dataset(truncate_dna_per_side: int = 1024) -> List[Dict[str, Any]]:
    """
    加载 KEGG 的 val 和 test 数据集。

    Args:
        truncate_dna_per_side: DNA 序列两端各截断多少碱基对

    Returns:
        格式化后的样本列表，每个样本包含 prompt、dna_sequences、answer 等字段
    """
    print("Loading KEGG val and test datasets from HuggingFace...")

    # 加载数据集的所有 split
    dataset = load_dataset('wanglab/kegg', 'default')
    test_dataset = dataset['test']
    val_dataset = dataset['val']
    # 合并 val 和 test：因为 KEGG 的 val 和 test 规模都较小
    test_val_dataset = concatenate_datasets([test_dataset, val_dataset])
    print(f"Loaded {len(test_val_dataset)} validation and test examples")

    # DNA 序列截断（控制长度）
    if truncate_dna_per_side > 0:
        test_val_dataset = test_val_dataset.map(
            truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side}
        )

    # 格式化为 DNA-LLM 模式（is_sft=False：不包含 assistant 回复，让模型自己生成）
    formatted_test_val_examples = []
    for example in test_val_dataset:
        formatted_example = format_kegg_for_dna_llm(example, is_sft=False)
        formatted_test_val_examples.append(formatted_example)

    print(f"Formatted {len(formatted_test_val_examples)} examples for DNA-LLM evaluation")
    return formatted_test_val_examples


def initialize_model(
    ckpt_dir: str,
    cache_dir: str,
    text_model_name: str = "Qwen/Qwen3-4B",
    dna_model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    max_length_dna: int = 1024,
    max_length_text: int = 512,
    gpu_memory_utilization: float = 0.4,
    max_model_len: int = 8192,
    dna_is_evo2: bool = False,
    dna_embedding_layer: str = None,
) -> DNALLMModel:
    """
    初始化 DNA-vLLM 推理模型。

    与训练时的 DNALLMModel 不同，这里的 DNALLMModel 来自 dna_vllm.py，
    内置了 vLLM LLM 引擎，不需要 Trainer 就能高效推理。

    关键参数：
    - gpu_memory_utilization: vLLM 占用 GPU 显存的比例（评估时设低一点）
    - max_model_len: 模型最大序列长度（prompt + 生成）
    - ckpt_dir: 训练好的 checkpoint 目录
    """
    print("Initializing DNA-vLLM model...")

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpt_dir = str(Path(ckpt_dir).expanduser())

    model = DNALLMModel(
        ckpt_dir=ckpt_dir,
        text_model_name=text_model_name,
        dna_model_name=dna_model_name,
        cache_dir=cache_dir,
        max_length_dna=max_length_dna,
        max_length_text=max_length_text,
        text_model_finetune=False,
        dna_model_finetune=False,
        dna_is_evo2=dna_is_evo2,
        dna_embedding_layer=dna_embedding_layer,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    print("Model initialized successfully!")
    return model


def evaluate_single_example(
    model: DNALLMModel,
    processor: DLProcessor,
    example: Dict[str, Any],
    generation_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    对单条样本进行推理评估。

    流程：
    1. 用 chat template 渲染 prompt 文本
    2. 通过 DLProcessor 处理文本和 DNA（替换 <|dna_pad|> 占位符）
    3. 调用 model.generate() 生成回复
    4. 从回复中提取 "Answer:" 后的内容
    5. 与 ground truth 比较判断是否正确

    Args:
        model: DNA-vLLM 推理模型
        processor: DLProcessor（文本+DNA 双处理器）
        example: 单条 KEGG 样本
        generation_kwargs: 生成参数（temperature, top_p, max_new_tokens 等）

    Returns:
        包含生成文本、预测答案、是否正确等信息的字典
    """
    # Step 1+2：渲染 prompt 并通过 processor 处理
    # maybe_apply_chat_template 将 prompt 结构化为文本
    prompts_text = [maybe_apply_chat_template(example, processor)["prompt"]]
    prepared = processor(
        text=prompts_text,
        batch_dna_sequences=[example["dna_sequences"]],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=model.max_length_text,
        max_length_dna=model.max_length_dna,
    )

    # Step 3：生成回复
    outputs = model.generate(
        input_ids=prepared["input_ids"],
        attention_mask=prepared["attention_mask"],
        dna_tokenized=prepared.get("dna_tokenized"),
        batch_idx_map=prepared.get("batch_idx_map"),
        **generation_kwargs
    )

    # Step 4：提取 "Answer:" 后的内容
    # 模型被训练为输出 "Answer: yes" 或 "Answer: no" 的格式
    generated_text = outputs[0] if outputs else ""

    predicted_answer = ""
    if "Answer:" in generated_text:
        answer_part = generated_text.split("Answer:")[-1].strip()
        predicted_answer = answer_part.lower()

    # 获取 ground truth
    ground_truth = example["answer"].strip().lower()

    # 清理标点符号（确保比较时不受标点干扰）
    for char in ".,!?\"'":
        predicted_answer = predicted_answer.replace(char, "")
        ground_truth = ground_truth.replace(char, "")

    # Step 5：判断是否正确（子串匹配）
    is_correct = ground_truth in predicted_answer
    print('predicted_answer:', predicted_answer, 'ground_truth:', ground_truth, 'is_correct:', is_correct)

    return {
        'prompts_text': prompts_text,
        "generated_text": generated_text,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "dna_sequences": example["dna_sequences"],
        "question": example["prompt"][0]["content"][-1]["text"]  # 提取原始问题文本
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    从评估结果计算分类指标。

    支持两种场景：
    - 二分类（2 个标签）：计算 accuracy, precision, recall, F1
    - 多分类（>2 个标签）：只计算 accuracy

    二分类混淆矩阵：
                         预测=正      预测=负
        实际=正          TP           FN
        实际=负          FP           TN

    precision = TP / (TP + FP)  — 预测为正的样本中有多少是真的
    recall    = TP / (TP + FN)  — 实际为正的样本中有多少被找到了
    F1        = 2*P*R / (P+R)   — precision 和 recall 的调和平均
    """
    total_examples = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])

    accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0

    # 获取所有唯一的 ground truth 标签
    all_answers = [r["ground_truth"] for r in results]
    unique_answers = list(set(all_answers))

    if len(unique_answers) == 2:
        # 二分类：计算完整的分类指标
        pos_label = unique_answers[0]
        neg_label = unique_answers[1]

        true_positives = sum(1 for r in results
                           if r["ground_truth"] == pos_label and r["predicted_answer"] == pos_label)
        false_positives = sum(1 for r in results
                            if r["ground_truth"] == neg_label and r["predicted_answer"] == pos_label)
        false_negatives = sum(1 for r in results
                            if r["ground_truth"] == pos_label and r["predicted_answer"] == neg_label)
        true_negatives = sum(1 for r in results
                           if r["ground_truth"] == neg_label and r["predicted_answer"] == neg_label)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "total_examples": total_examples,
            "correct_predictions": correct_predictions,
            "positive_label": pos_label,
            "negative_label": neg_label
        }
    else:
        # 多分类：只返回准确率
        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct_predictions": correct_predictions,
            "unique_labels": unique_answers
        }


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_dir: str,
    model_name: str = "dna_vllm"
) -> None:
    """
    保存评估结果到文件。

    输出三个文件：
    - {model_name}_kegg_eval_results_{timestamp}.json  — 详细结果（每条样本的预测）
    - {model_name}_kegg_eval_metrics_{timestamp}.json  — 汇总指标
    - {model_name}_kegg_eval_results_{timestamp}.csv   — CSV 格式便于分析
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果（JSON）
    results_file = os.path.join(output_dir, f"{model_name}_kegg_eval_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")

    # 保存指标（JSON）
    metrics_file = os.path.join(output_dir, f"{model_name}_kegg_eval_metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    # 保存结果表格（CSV）
    csv_file = os.path.join(output_dir, f"{model_name}_kegg_eval_results_{timestamp}.csv")
    df_data = []
    for i, result in enumerate(results):
        df_data.append({
            "example_id": i,
            "question": result["question"],
            "predicted_answer": result["predicted_answer"],
            "ground_truth": result["ground_truth"],
            "is_correct": result["is_correct"],
            "generated_text": result["generated_text"]
        })

    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    print(f"Results CSV saved to: {csv_file}")

    # 打印评估摘要
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Positive label: {metrics['positive_label']}")
        print(f"Negative label: {metrics['negative_label']}")

    print("="*80)


def main():
    """评估主函数：加载数据 → 初始化模型 → 逐条推理 → 计算指标 → 保存结果"""
    parser = argparse.ArgumentParser(description="Evaluate DNA-vLLM on KEGG test set")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to the checkpoint directory")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Cache directory for models")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-4B",
                        help="Name of the text model")
    parser.add_argument("--dna_model_name", type=str,
                        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                        help="Name of the DNA model")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max_length_dna", type=int, default=1024,
                        help="Maximum length of the DNA sequence")
    parser.add_argument("--max_length_text", type=int, default=1024,
                        help="Maximum length of the text sequence")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate (None for all)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature for generation (0=贪婪解码)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p for generation")
    parser.add_argument("--max_new_tokens", type=int, default=800,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--dna_is_evo2", type=bool, default=False,
                        help="Whether the DNA model is Evo2")
    parser.add_argument("--dna_embedding_layer", type=str, default=None,
                        help="Name of the layer to use for the Evo2 model")
    parser.add_argument("--truncate_dna_per_side", type=int, default=0,
                        help="Number of base pairs to truncate from each end of the DNA sequence")
    args = parser.parse_args()

    print("="*80)
    print("DNA-vLLM KEGG Evaluation Script")
    print("="*80)
    print(f"Checkpoint directory: {args.ckpt_dir}")
    print(f"Text model: {args.text_model_name}")
    print(f"DNA model: {args.dna_model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max examples: {args.max_examples if args.max_examples else 'All'}")
    print("="*80)

    try:
        # 加载数据集
        test_examples = load_kegg_test_dataset(truncate_dna_per_side=args.truncate_dna_per_side)

        # 可选：限制评估样本数（快速测试用）
        if args.max_examples:
            test_examples = test_examples[:args.max_examples]
            print(f"Limited to {len(test_examples)} examples")

        # 初始化模型（vLLM 版本）
        model = initialize_model(
            ckpt_dir=args.ckpt_dir,
            cache_dir=args.cache_dir,
            text_model_name=args.text_model_name,
            dna_model_name=args.dna_model_name,
            max_length_dna=args.max_length_dna,
            max_length_text=args.max_length_text,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dna_is_evo2=args.dna_is_evo2,
            dna_embedding_layer=args.dna_embedding_layer,
        )

        # 初始化 DLProcessor（文本+DNA 双处理器）
        processor = DLProcessor(
            tokenizer=model.text_tokenizer,
            dna_tokenizer=model.dna_tokenizer,
        )

        # 生成参数
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "stop": ["<|im_end|>"],  # Qwen3 的对话结束标记
        }

        print(f"\nStarting evaluation of {len(test_examples)} examples...")
        print("Generation parameters:")
        for key, value in generation_kwargs.items():
            print(f"  {key}: {value}")

        # 逐条评估
        results = []
        for i, example in enumerate(tqdm(test_examples, desc="Evaluating")):
            result = evaluate_single_example(
                model=model,
                processor=processor,
                example=example,
                generation_kwargs=generation_kwargs
            )
            results.append(result)

            # 每 10 条打印一次进度
            if (i + 1) % 10 == 0:
                correct_so_far = sum(1 for r in results if r["is_correct"])
                accuracy_so_far = correct_so_far / (i + 1)
                print(f"Progress: {i+1}/{len(test_examples)} | Accuracy so far: {accuracy_so_far:.4f}")

        # 计算指标
        print("\nCalculating metrics...")
        metrics = calculate_metrics(results)

        # 保存结果
        save_results(results, metrics, args.output_dir)

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
