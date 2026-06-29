"""
GRPO 强化学习训练脚本 — Variant Effect 专用版本
================================================

与 SFT (train_dna_qwen_vegg.py) 的关系:
  - SFT 训练好 checkpoint → GRPO 从 checkpoint 热启动
  - GRPO 通过奖励信号进一步优化模型的推理能力

GRPO 原理 (DeepSeek, 2024):
  1. 对同一 prompt 生成 G 个回复 (group)
  2. 用 reward 函数给每个回复打分
  3. 组内归一化: advantage = reward - mean(group_rewards)
  4. 用 PPO clipped loss 更新策略

两阶段:
  --stage 1: Pathogenic vs Benign
  --stage 2: GOF vs LOF

用法:
  # 阶段一
  python train_grpo_vegg.py --stage 1 \
      --sft_checkpoint /path/to/stage1.ckpt \
      --output_dir /path/to/output

  # 阶段二
  python train_grpo_vegg.py --stage 2 \
      --sft_checkpoint /path/to/stage2.ckpt \
      --output_dir /path/to/output
"""

import gc
import os
import pathlib
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from packaging import version
from transformers import AutoTokenizer, GenerationConfig

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import ModelConfig, ScriptArguments, TrlParser

from bioreason.models.dna_llm import DNALLMModel, get_target_modules
from bioreason.dna_modules import NucleotideDNAModule
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import qwen_dna_collate_fn
from bioreason.dataset.variant_effect import (
    clean_variant_effect_example,
    get_format_variant_effect_function,
)
from bioreason.trainer import DNALLMGRPOTrainer, DNALLMGRPOConfig
from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
register_evo2_tokenizer()

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint 保存回调 (PyTorch 格式，兼容 DNA-LLM 非标模块)
# ══════════════════════════════════════════════════════════════════════════════

class SaveWithPyTorchCallback(TrainerCallback):
    """用 PyTorch 原生 save 替代 safetensors，兼容 DNA-LLM 的自定义模块。"""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        model = kwargs.get("model")
        unwrapped_model = model.module if hasattr(model, "module") else model
        torch.save(unwrapped_model.state_dict(), checkpoint_path)
        if hasattr(unwrapped_model, "text_model") and hasattr(unwrapped_model.text_model, "config"):
            unwrapped_model.text_model.config.save_pretrained(checkpoint_folder)
        print(f"Saved model checkpoint to {checkpoint_folder}")
        control.should_save = False
        return control


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载: CSV → 基因感知分割 → GRPO 格式化
# ══════════════════════════════════════════════════════════════════════════════

def load_variant_effect_for_grpo(
    stage: int,
    text_model_name: str,
    truncate_dna_per_side: int = 0,
) -> Dict[str, Dataset]:
    """
    加载 Variant Effect 数据，做基因感知分割，格式化为 GRPO 模式。

    流程:
    1. 加载 CSV (Stage1 或 Stage2)
    2. 从 question 中提取基因名
    3. 基因分类 (纯GOF/纯LOF/混合)
    4. 分层 8:1:1 分割
    5. 格式化为 GRPO 模式 (is_sft=False)

    Returns:
        {"train": Dataset, "val": Dataset, "test": Dataset}
    """
    if stage == 1:
        data_file = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage1_Binary.csv"
    else:
        data_file = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage2_GOF_LOF.csv"

    dataset = load_dataset("csv", data_files=data_file)
    raw_data = dataset["train"]
    print(f"加载数据: {len(raw_data)} 条 (Stage {stage})")

    # Step 1: 提取基因名
    def _extract_gene(example):
        q = example.get("question", "")
        m = re.search(r"- Gene: (\S+)", q)
        return {"_gene": m.group(1) if m else ""}
    raw_data = raw_data.map(_extract_gene, load_from_cache_file=False)

    # Step 2: 统计每个基因的标签种类
    gene_to_labels = defaultdict(set)
    gene_to_type = {}
    for example in raw_data:
        gene = example.get("_gene", "")
        if not gene:
            continue
        ans = example.get("answer", "").strip().lower()
        gene_to_labels[gene].add(ans)
        gene_to_type[gene] = example.get("gene_type", "shared")

    # Step 3: 基因分类
    if stage == 1:
        pure_path, pure_benign, mixed_s1 = [], [], []
        for gene, labels in gene_to_labels.items():
            has_path = any("pathogenic" in l for l in labels)
            has_benign = any("benign" in l for l in labels)
            if has_path and not has_benign:
                pure_path.append(gene)
            elif has_benign and not has_path:
                pure_benign.append(gene)
            else:
                mixed_s1.append(gene)
        buckets = [pure_path, pure_benign, mixed_s1]
    else:
        pure_gof, pure_lof, mixed_s2, other_s2 = [], [], [], []
        for gene, labels in gene_to_labels.items():
            has_gof = any("gain-of-function" in l for l in labels)
            has_lof = any("loss-of-function" in l for l in labels)
            if has_gof and not has_lof:
                pure_gof.append(gene)
            elif has_lof and not has_gof:
                pure_lof.append(gene)
            elif has_gof and has_lof:
                mixed_s2.append(gene)
            else:
                other_s2.append(gene)
        buckets = [pure_gof, pure_lof, mixed_s2, other_s2]

    # Step 4: 分层 8:1:1
    def _split_genes(gene_list):
        genes = sorted(set(gene_list))
        random.Random(42).shuffle(genes)
        n = len(genes)
        t_end = int(n * 0.8)
        v_end = int(n * 0.9)
        return genes[:t_end], genes[t_end:v_end], genes[v_end:]

    train_genes, val_genes, test_genes = set(), set(), set()
    for bucket in buckets:
        t, v, te = _split_genes(bucket)
        train_genes.update(t)
        val_genes.update(v)
        test_genes.update(te)

    # Stage 2: lof_only 基因全部进 train
    if stage == 2:
        for gene, gtype in gene_to_type.items():
            if gtype == "lof_only":
                train_genes.add(gene)
                val_genes.discard(gene)
                test_genes.discard(gene)

    print(f"基因分割: train={len(train_genes)}, val={len(val_genes)}, test={len(test_genes)}")

    # Step 5: 按基因过滤 + 格式化
    format_fn = get_format_variant_effect_function("dna-llm", is_sft=False)

    def _filter_and_format(data, gene_set, split_name):
        filtered = data.filter(lambda x: x.get("_gene", "") in gene_set)
        formatted = filtered.map(format_fn, load_from_cache_file=False)
        print(f"  {split_name}: {len(formatted)} 条")
        return formatted

    result = {
        "train": _filter_and_format(raw_data, train_genes, "train"),
        "val": _filter_and_format(raw_data, val_genes, "val"),
        "test": _filter_and_format(raw_data, test_genes, "test"),
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GRPO 参数配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GRPOModelConfig(ModelConfig):
    text_model_name: str = field(default="Qwen/Qwen3-4B")
    dna_model_name: str = field(default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    cache_dir: str = field(default=None)
    max_length_text: int = field(default=2048)
    max_length_dna: int = field(default=512)
    sft_checkpoint: str = field(default=None, metadata={"help": "SFT Lightning checkpoint 路径"})
    # SFT 阶段的 LoRA 配置 (用于正确加载 checkpoint)
    sft_lora_r: int = field(default=64, metadata={"help": "SFT checkpoint 的 LoRA rank"})
    sft_lora_alpha: int = field(default=128, metadata={"help": "SFT checkpoint 的 LoRA alpha"})
    sft_lora_dropout: float = field(default=0.0)
    # GRPO 阶段的 LoRA 配置
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0)
    lora_modules_to_save: Optional[List[str]] = field(default="embed_tokens")
    dna_model_finetune: bool = False
    dna_projection_finetune: bool = True
    dna_is_evo2: bool = False
    dna_embedding_layer: str = field(default=None)
    truncate_dna_per_side: int = field(default=256)
    stage: int = field(default=1, metadata={"help": "1=Pathogenic vs Benign, 2=GOF vs LOF"})


@dataclass
class GRPOScriptArguments(ScriptArguments):
    dataset_name: str = field(default="variant_effect_coding")
    data_file_paths: str = field(default=None)
    arrow_cache_dir: str = field(default=None)
    val_split_ratio: float = field(default=0.0)
    reward_funcs: List[str] = field(
        default_factory=lambda: ["xmlcount", "soft_format", "strict_format", "concise", "correctness"],
        metadata={"help": "Reward 函数列表"}
    )


# Reward 函数注册表
reward_funcs_registry = {
    "xmlcount": NucleotideDNAModule.xmlcount_reward_func,
    "soft_format": NucleotideDNAModule.soft_format_reward_func,
    "strict_format": NucleotideDNAModule.strict_format_reward_func,
    "concise": NucleotideDNAModule.concise_reward_func,
    "correctness": NucleotideDNAModule.correctness_reward_func,
}


# ══════════════════════════════════════════════════════════════════════════════
# 模型准备: 加载 SFT checkpoint → 合并 LoRA → 添加 GRPO LoRA
# ══════════════════════════════════════════════════════════════════════════════

def _load_sft_checkpoint(model: DNALLMModel, ckpt_path: str, sft_lora_r: int, sft_lora_alpha: int, sft_lora_dropout: float):
    """
    加载 SFT Lightning checkpoint 并合并 LoRA 权重到 base model。

    流程:
    1. 加载 .ckpt 文件
    2. 提取 state_dict，去除 "model." 前缀
    3. 检测是否有 LoRA 权重
    4. 如有 LoRA: 先添加匹配的 LoRA 结构 → 加载权重 → merge_and_unload
    5. 如无 LoRA: 直接加载权重
    """
    print(f"Loading SFT checkpoint from {ckpt_path}")
    if os.path.isdir(ckpt_path):
        # DeepSpeed ZeRO checkpoint (sharded directory)
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        raw_state = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
        elif "module" in checkpoint:
            raw_state = checkpoint["module"]
        else:
            raw_state = checkpoint

    # 去除 Lightning 的 "model." 前缀
    state_dict = {}
    for k, v in raw_state.items():
        if k.startswith("model."):
            k = k[6:]
        elif k.startswith("_forward_module."):
            k = k[len("_forward_module."):]
        state_dict[k] = v

    has_lora = any("lora" in k for k in state_dict)
    print(f"  Checkpoint keys: {len(state_dict)}, has_lora={has_lora}")

    if has_lora:
        # 先用 SFT 的 LoRA 配置包装模型 (使 key 能匹配)
        print(f"  Adding SFT LoRA structure (r={sft_lora_r}, alpha={sft_lora_alpha})...")
        target_modules = get_target_modules(model)
        sft_lora_config = LoraConfig(
            r=sft_lora_r,
            lora_alpha=sft_lora_alpha,
            lora_dropout=sft_lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.text_model = prepare_model_for_kbit_training(model.text_model)
        model.text_model = get_peft_model(model.text_model, sft_lora_config)

        # 加载权重
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {len(state_dict) - len(missing)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 0:
            print(f"  Sample missing: {missing[:5]}")
        if len(unexpected) > 0:
            print(f"  Sample unexpected: {unexpected[:5]}")

        # 合并 SFT LoRA 到 base weights
        print("  Merging SFT LoRA into base model...")
        model.text_model = model.text_model.merge_and_unload()
        print("  Merge complete.")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded non-LoRA weights: {len(missing)} missing, {len(unexpected)} unexpected")

    return model


def _prep_for_grpo(
    model: DNALLMModel,
    training_args,
    dna_model_finetune: bool = False,
    dna_projection_finetune: bool = True,
) -> LoraConfig:
    """
    配置 GRPO 训练的各组件冻结/训练状态，并添加 GRPO 专用 LoRA。

    训练策略:
    - DNA Encoder:      冻结 (预训练权重)
    - DNA Projection:    可训练 (桥接层，适应新奖励信号)
    - Text LLM:          LoRA 微调 (高效适应)
    """
    # DNA encoder: 冻结
    if hasattr(model, 'dna_is_evo2') and model.dna_is_evo2:
        dna_model = model.dna_model.model
    else:
        dna_model = model.dna_model

    if dna_model_finetune:
        dna_model.train()
        for param in dna_model.parameters():
            param.requires_grad = True
        print("DNA model is trainable")
    else:
        dna_model.eval()
        for param in dna_model.parameters():
            param.requires_grad = False
        print("DNA model is frozen")

    # DNA projection: 可训练
    if dna_projection_finetune:
        model.dna_projection.train()
        for param in model.dna_projection.parameters():
            param.requires_grad = True
        print("DNA projection is trainable")
    else:
        model.dna_projection.eval()
        for param in model.dna_projection.parameters():
            param.requires_grad = False
        print("DNA projection is frozen")

    # Text LLM: 添加 GRPO 专用 LoRA
    if training_args.lora_r == 0:
        model.text_model.train()
        for param in model.text_model.parameters():
            param.requires_grad = True
        return None
    else:
        target_modules = get_target_modules(model)
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.text_model = prepare_model_for_kbit_training(model.text_model)
        model.text_model = get_peft_model(model.text_model, lora_config)
        model.text_model.train()
        print(f"GRPO LoRA added: r={training_args.lora_r}, alpha={training_args.lora_alpha}")
        return lora_config


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # ── 1. 初始化 DNA-LLM 模型 ──
    print("=" * 60)
    print(f"Initializing DNA-LLM for GRPO (Stage {model_args.stage})...")
    model = DNALLMModel(
        text_model_name=model_args.text_model_name,
        dna_model_name=model_args.dna_model_name,
        cache_dir=model_args.cache_dir,
        max_length_text=model_args.max_length_text,
        max_length_dna=model_args.max_length_dna,
        text_model_finetune=True,
        dna_model_finetune=model_args.dna_model_finetune,
        dna_is_evo2=model_args.dna_is_evo2,
        dna_embedding_layer=model_args.dna_embedding_layer,
        device="cuda",
    ).to("cuda")
    model.text_model.config.use_cache = False

    # ── 2. 加载 SFT checkpoint 并合并 LoRA ──
    if model_args.sft_checkpoint is not None:
        # vLLM 用基础模型初始化，训练时通过 _move_model_to_vllm 同步 SFT 权重
        training_args.vllm_ckpt = model_args.text_model_name
        _load_sft_checkpoint(
            model,
            model_args.sft_checkpoint,
            sft_lora_r=model_args.sft_lora_r,
            sft_lora_alpha=model_args.sft_lora_alpha,
            sft_lora_dropout=model_args.sft_lora_dropout,
        )
    else:
        print("WARNING: No SFT checkpoint provided, starting from pretrained model!")

    # ── 3. 添加 GRPO LoRA ──
    _prep_for_grpo(
        model,
        model_args,
        dna_model_finetune=model_args.dna_model_finetune,
        dna_projection_finetune=model_args.dna_projection_finetune,
    )

    model = model.to(training_args.device)

    # ── 4. Reward 函数 ──
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print(f"Reward functions: {script_args.reward_funcs}")
    print(f"Vocab size: {len(model.text_tokenizer)}, dna_pad_id: {model.text_tokenizer.convert_tokens_to_ids('<|dna_pad|>')}")

    # ── 5. 加载数据 ──
    print(f"\nLoading Variant Effect data (Stage {model_args.stage})...")
    dataset = load_variant_effect_for_grpo(
        stage=model_args.stage,
        text_model_name=model_args.text_model_name,
        truncate_dna_per_side=model_args.truncate_dna_per_side,
    )

    # ── 6. 创建 GRPO Trainer ──
    custom_save_callback = SaveWithPyTorchCallback()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {total_trainable:,} / {total_params:,} ({100 * total_trainable / total_params:.2f}%)")

    trainer = DNALLMGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        dna_module=NucleotideDNAModule(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        peft_config=None,
        callbacks=[custom_save_callback],
        processing_class=model.processor,
    )

    training_args.save_safetensors = False

    # 打印可训练参数
    print("=" * 50)
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [TRAINABLE]: {name} | {param.shape}")
    print("=" * 50)

    # ── 7. 处理 checkpoint 恢复 ──
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if resume_from_checkpoint in ("True", "true"):
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = str(max(checkpoints, key=os.path.getmtime))
            print(f"Resuming from: {resume_from_checkpoint}")
        else:
            print("No checkpoint found, starting fresh.")
            resume_from_checkpoint = None
    elif resume_from_checkpoint:
        print(f"Resuming from: {resume_from_checkpoint}")

    # ── 8. 开始训练 ──
    print("\n" + "=" * 60)
    print(f"Starting GRPO training (Stage {model_args.stage})...")
    print("=" * 60)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo-vegg")

    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    training_args.save_safetensors = False
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
