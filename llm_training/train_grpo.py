"""
GRPO (Group Relative Policy Optimization) 训练入口脚本。
=============================================================================
GRPO 是 DeepSeek 提出的强化学习微调方法，核心思想是：
  1. 对同一个 prompt 生成多个回复（group）
  2. 用 reward 函数给每个回复打分
  3. 组内归一化：每个回复的 advantage = 自己的reward - 组内平均reward
  4. 用 PPO 风格的 clipped objective 更新策略

与 SFT 的区别：
  - SFT: 直接最大化正确答案的概率（监督学习）
  - GRPO: 通过奖励信号探索更好的生成策略（强化学习）

训练流程：
  1. 加载 DNA-LLM 模型（可选：加载 SFT checkpoint 作为初始化）
  2. 配置 LoRA 微调（只训练 text_model 的 LoRA 适配器 + dna_projection）
  3. 加载 KEGG 数据集
  4. 用 DNALLMGRPOTrainer 进行 GRPO 训练
     - 生成阶段：vLLM 高效生成多个回复
     - 评分阶段：多个 reward 函数打分（正确性、格式、简洁性等）
     - 更新阶段：计算 advantage 并用 PPO loss 更新模型
=============================================================================
"""

import os

import pathlib
from typing import List, Optional
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from datasets import load_dataset

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel

from trl import ModelConfig, ScriptArguments, TrlParser

from bioreason.models.dna_llm import DNALLMModel, get_target_modules
from bioreason.dna_modules import NucleotideDNAModule
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import format_kegg_for_dna_llm
from bioreason.trainer import DNALLMGRPOTrainer, DNALLMGRPOConfig
from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
register_evo2_tokenizer()

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SaveWithPyTorchCallback(TrainerCallback):
    """
    自定义保存回调：用 PyTorch 原生 save 替代 safetensors 格式保存 checkpoint。

    为什么需要这个：
    - DNA-LLM 模型包含非标准的子模块（DNA encoder、projection layer）
    - safetensors 对这些自定义结构的兼容性不好
    - PyTorch 原生保存更可靠，尤其是有 LoRA 适配器时
    """
    def on_save(self, args, state, control, **kwargs):
        # 构建 checkpoint 目录路径
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        os.makedirs(checkpoint_folder, exist_ok=True)

        # 用 PyTorch 保存（而非 safetensors）
        checkpoint_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        model = kwargs.get("model")

        # 获取未包装的模型（去掉 accelerator/DeepSpeed 的包装层）
        unwrapped_model = model.module if hasattr(model, "module") else model

        # 保存完整 state_dict（包含 LoRA 权重和 projection 层权重）
        torch.save(unwrapped_model.state_dict(), checkpoint_path)

        # 保存 text_model 的 config（HuggingFace 加载时需要）
        if hasattr(unwrapped_model, "text_model"):
            if hasattr(unwrapped_model.text_model, "config"):
                unwrapped_model.text_model.config.save_pretrained(checkpoint_folder)
            elif hasattr(unwrapped_model.text_model, "base_model") and hasattr(unwrapped_model.text_model.base_model, "config"):
                unwrapped_model.text_model.base_model.config.save_pretrained(checkpoint_folder)

        # 打印保存信息
        print(f"Saved model checkpoint to {checkpoint_folder}")
        lora_params = [k for k in unwrapped_model.state_dict().keys() if "lora" in k]
        print(f"Checkpoint contains {len(lora_params)} LoRA parameters")

        # 告诉 Trainer 我们已经自己保存了，不需要再保存
        control.should_save = False
        return control


def get_kegg_questions(truncate_dna_per_side: int = 0) -> Dataset:
    """
    加载 KEGG 数据集并格式化为 DNA-LLM 格式。

    Args:
        truncate_dna_per_side: DNA 序列两端截断的碱基对数，0 表示不截断

    Returns:
        格式化后的 HuggingFace Dataset（包含 train/val/test 三个 split）
    """
    data = load_dataset('wanglab/kegg', 'default')

    # 截断 DNA 序列以控制长度
    if truncate_dna_per_side > 0:
        data = data.map(truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side})

    # 格式化为 DNA-LLM 模式（is_sft=False：不添加 assistant 回复，因为 GRPO 需要模型自己生成）
    data = data.map(format_kegg_for_dna_llm, fn_kwargs={"is_sft": False})

    return data


@dataclass
class GRPOModelConfig(ModelConfig):
    """
    GRPO 训练的模型配置（扩展 TRL 的 ModelConfig）。

    新增字段说明：
    - text_model_name: 文本 LLM 的名称或路径（如 Qwen3-4B）
    - dna_model_name:  DNA 编码器的名称或路径（如 NucleotideTransformer）
    - sft_checkpoint:  可选，SFT 阶段训练好的 checkpoint 路径
    - lora_r/alpha/dropout: LoRA 配置参数
    - dna_is_evo2: 是否使用 Evo2 作为 DNA 编码器（需要特殊加载逻辑）
    """
    text_model_name: str = field(default="Qwen/Qwen3-4B", metadata={"help": "Model checkpoint for weights initialization."})
    dna_model_name: str = field(default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", metadata={"help": "Model checkpoint for weights initialization."})
    cache_dir: str = field(default=None, metadata={"help": "Path to model cache directory."})
    max_length_text: int = field(default=1024, metadata={"help": "Maximum length of text sequences."})
    max_length_dna: int = field(default=1024, metadata={"help": "Maximum length of DNA sequences, in groups of 6 nucleotides."})
    sft_checkpoint: str = field(default=None, metadata={"help": "Path to the checkpoint for SFT."})
    lora_r: int = field(default=16, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0, metadata={"help": "LoRA dropout."})
    lora_modules_to_save: Optional[List[str]] = field(
        default="embed_tokens",
        metadata={"help": "Model layers to unfreeze & train."},
    )
    dna_model_finetune: bool = False
    dna_projection_finetune: bool = True
    peft_ckpt: bool = False
    dna_is_evo2: bool = field(default=False, metadata={"help": "Whether the DNA model is Evo2."})
    dna_embedding_layer: str = field(default=None, metadata={"help": "Evo2 layer name to extract embeddings from (required when dna_is_evo2=True)."})
    truncate_dna_per_side: int = field(default=0, metadata={"help": "Number of base pairs to truncate from each end of the DNA sequence. If 0, no truncation is applied."})


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    GRPO 训练的脚本参数。
    """
    dataset_name: str = field(default="wanglab/kegg", metadata={"help": "Dataset name with default."})
    full_ckpt: str = field(default=None, metadata={"help": "Path to full checkpoint to load"})
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: List[str] = field(
        # 默认使用全部 5 个 reward 函数
        default_factory=lambda: ["xmlcount", "soft_format", "strict_format", "concise", "correctness"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'xmlcount', 'soft_format', 'strict_format', 'concise', 'correctness', 'depth'"},
    )


# Reward 函数注册表：字符串名 → 实际函数
# 每个 reward 函数接收 prompts 和 completions，返回一个分数列表
reward_funcs_registry = {
    "xmlcount": NucleotideDNAModule.xmlcount_reward_func,        # XML 标签计数是否正确
    "soft_format": NucleotideDNAModule.soft_format_reward_func,  # 格式软检查（宽松）
    "strict_format": NucleotideDNAModule.strict_format_reward_func,  # 格式严格检查
    "concise": NucleotideDNAModule.concise_reward_func,          # 简洁性奖励
    "correctness": NucleotideDNAModule.correctness_reward_func,  # 答案正确性
}


def get_vlm_module(text_model_name):
    """
    根据文本模型名称返回对应的 DNA 模块类。

    DNA 模块负责：
    - 准备 prompt（应用 chat template）
    - 提供 reward 函数
    - 管理非 generate 参数
    """
    if any(mini_name in text_model_name.lower() for mini_name in ["qwen", "smol"]):
        return NucleotideDNAModule
    else:
        raise ValueError(f"Unsupported model: {text_model_name}")


def _prep_for_training(
    model: DNALLMModel,
    training_args,
    dna_model_finetune: bool = False,
    dna_projection_finetune: bool = True
    ) -> Optional[LoraConfig]:
    """
    配置模型各组件的训练/冻结状态，并设置 LoRA。

    组件训练策略（GRPO 阶段）：
    ┌─────────────────────┬──────────┬──────────────────────────────┐
    │ 组件                 │ 状态     │ 原因                          │
    ├─────────────────────┼──────────┼──────────────────────────────┤
    │ DNA Encoder (NT/Evo)│ 冻结     │ 预训练权重已经很好，不需微调    │
    │ DNA Projection       │ 可训练   │ 桥梁层，需要适应新的奖励信号    │
    │ Text LLM             │ LoRA     │ 用 LoRA 高效微调，不破坏预训练  │
    │                      │   微调    │ 知识                         │
    └─────────────────────┴──────────┴──────────────────────────────┘

    Returns:
        LoRA 配置对象，如果 lora_r=0 则返回 None（全量微调模式）
    """
    # ── DNA 编码器：通常冻结 ──
    # Evo2 和 HuggingFace 模型的访问路径不同
    if hasattr(model, 'dna_is_evo2') and model.dna_is_evo2:
        dna_model = model.dna_model.model
    else:
        dna_model = model.dna_model

    if dna_model_finetune:
        dna_model.train()
        print("DNA model is training")
        for param in dna_model.parameters():
            param.requires_grad = True
    else:
        dna_model.eval()
        print("DNA model is eval")
        for param in dna_model.parameters():
            param.requires_grad = False

    # ── DNA Projection：通常可训练 ──
    if dna_projection_finetune:
        model.dna_projection.train()
        print("DNA projection is training")
        for param in model.dna_projection.parameters():
            param.requires_grad = True
    else:
        model.dna_projection.eval()
        print("DNA projection is eval")
        for param in model.dna_projection.parameters():
            param.requires_grad = False

    # ── Text LLM：LoRA 微调 ──
    if training_args.lora_r == 0:
        # lora_r=0 表示全量微调（不推荐，显存消耗大）
        model.text_model.train()
        print("Text model is training")
        for param in model.text_model.parameters():
            param.requires_grad = True
        return None
    else:
        # 获取所有需要应用 LoRA 的线性层名称
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

        # 准备模型并进行 LoRA 包装
        model.text_model = prepare_model_for_kbit_training(model.text_model)
        model.text_model = get_peft_model(model.text_model, lora_config)
        model.text_model.train()

        return lora_config


def main(script_args, training_args, model_args):
    """
    GRPO 训练主函数。

    流程：
    1. 初始化 DNA-LLM 模型
    2. 加载 SFT checkpoint（两种方式：PEFT 目录 或 PyTorch state_dict 文件）
    3. 配置训练参数（LoRA/冻结/全量微调）
    4. 加载数据集和 reward 函数
    5. 创建 DNALLMGRPOTrainer 并开始训练
    """
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # ── 步骤1：初始化 DNA-LLM 模型 ──
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

    model.text_model.config.use_cache = False  # gradient checkpointing 需要关闭 cache

    # ── 步骤2：加载 SFT checkpoint（如果提供了）──
    # 这很关键：GRPO 从 SFT 模型开始训练，比从零开始的预训练模型效果更好
    # 因为 SFT 模型已经学会了基本的回答格式和推理模式
    if model_args.sft_checkpoint is not None:
        training_args.vllm_ckpt = model_args.sft_checkpoint
        print(f"Loading SFT checkpoint from {model_args.sft_checkpoint}")

        is_directory = os.path.isdir(model_args.sft_checkpoint)
        print(f"model_args.peft_ckpt: {model_args.peft_ckpt}")

        if is_directory and model_args.peft_ckpt:
            # 情况A：PEFT 格式的目录 checkpoint
            # 先用 from_pretrained 加载基础模型，再加载 PEFT 适配器
            # 最后 merge_and_unload 把 LoRA 权重合并到基础模型（方便 vLLM 使用）
            print(f"Loading tokenizer from PEFT checkpoint: {model_args.sft_checkpoint}")
            model.text_tokenizer = AutoTokenizer.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )

            model.text_model = AutoModelForCausalLM.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )

            print("Loading as PEFT checkpoint directory")
            model.text_model = PeftModel.from_pretrained(
                model.text_model,
                model_args.sft_checkpoint,
                is_trainable=True
            )

            print("Loaded LoRA adapters:", model.text_model.active_adapter)

            # 将 SFT 阶段的 LoRA 权重合并到基础模型
            # 原因：GRPO 阶段会重新添加新的 LoRA，旧的 LoRA 需要先固化
            print("Merging SFT LoRA weights into base model...")
            model.text_model = model.text_model.merge_and_unload()
            print("Successfully merged SFT knowledge into base model")

        elif is_directory and not model_args.peft_ckpt:
            # 情况B：标准 HuggingFace 格式的目录 checkpoint（非 PEFT）
            print(f"Loading tokenizer from checkpoint: {model_args.sft_checkpoint}")
            model.text_tokenizer = AutoTokenizer.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )

            model.text_model = AutoModelForCausalLM.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )
            print(f"CALLING FIRST PREP_FOR_TRAINING with dna_model_finetune: {getattr(model_args, 'dna_model_finetune', False)}, dna_projection_finetune: {getattr(model_args, 'dna_projection_finetune', False)}")
            _ = _prep_for_training(
                model,
                model_args,
                dna_model_finetune=getattr(model_args, "dna_model_finetune", False),
                dna_projection_finetune=getattr(model_args, "dna_projection_finetune", False)
            )
            print("model.text_model after loading", model.text_model)

            print("Successfully loaded SFT checkpoint")

        else:
            # 情况C：PyTorch state_dict 文件（.pt/.bin）
            # 需要手动处理 key 前缀映射，兼容不同的保存格式
            print("Loading as PyTorch state dict file")
            checkpoint = torch.load(model_args.sft_checkpoint)

            # 标准化 key：去掉 DDP/FSDP 前缀
            def new_key(k):
                if k.startswith("=model."): return k[6:]
                elif k.startswith("_forward_module."): return k[len("_forward_module."):]
                else: return k

            # 兼容多种 checkpoint 格式
            if "state_dict" in checkpoint:
                magic = {new_key(k): v for k, v in checkpoint["state_dict"].items()}
            elif "module" in checkpoint:
                magic = {new_key(k): v for k, v in checkpoint["module"].items()}
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                print("Detected direct state dict format")
                magic = {new_key(k): v for k, v in checkpoint.items()}
            else:
                raise ValueError(f"Unsupported checkpoint format: {model_args.sft_checkpoint}")

            # 检测是否包含 LoRA 权重
            lora_prefix = False
            for key in magic.keys():
                if "lora" in key:
                    lora_prefix = True
                    break

            if lora_prefix:
                # LoRA checkpoint：先初始化 LoRA 结构，再加载权重
                print("Detected LoRA weights in state dict")
                print(f"CALLING SECOND PREP_FOR_TRAINING with dna_model_finetune: {getattr(model_args, 'dna_model_finetune', False)}")
                _prep_for_training(model, model_args,
                    dna_model_finetune=getattr(model_args, "dna_model_finetune", False),
                    dna_projection_finetune=getattr(model_args, "dna_projection_finetune", False))

                # 诊断信息
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(magic.keys())
                print(f"Model has {len(model_keys)} keys")
                print(f"Checkpoint has {len(checkpoint_keys)} keys")

                # 智能 key 映射：处理 PEFT 包装导致的前缀差异
                new_magic = {}
                for k, v in magic.items():
                    if "base_model.model" in k and k not in model_keys:
                        new_k = k.replace("text_model.base_model.model", "text_model")
                        if new_k in model_keys:
                            new_magic[new_k] = v
                            continue

                    if k.startswith("text_model.") and k not in model_keys:
                        new_k = "text_model.base_model.model." + k[len("text_model."):]
                        if new_k in model_keys:
                            new_magic[new_k] = v
                            continue

                    new_magic[k] = v

                magic = new_magic
                print(f"After key mapping: {len(magic)} keys")

                result = model.load_state_dict(magic, strict=False)

                if len(result.unexpected_keys) > 0:
                    print(f"Sample unexpected keys: {result.unexpected_keys[:5]}")
                if len(result.missing_keys) > 0:
                    print(f"Sample missing keys: {result.missing_keys[:5]}")

                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")
            else:
                # 标准 checkpoint：先加载权重，再设置 LoRA
                print("Standard weights detected - remapping keys")
                magic = {k.replace("text_model", "text_model.base_model.model"): v for k, v in magic.items()}

                # 修复 shared memory tensors 问题（lm_head 与 embedding 共享权重时）
                for key in list(magic.keys()):
                    if 'lm_head.weight' in key:
                        magic[key] = magic[key].clone()

                result = model.load_state_dict(magic, strict=False)
                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")

                print(f"CALLING THIRD PREP_FOR_TRAINING with dna_model_finetune: {getattr(model_args, 'dna_model_finetune', False)}")
                _ = _prep_for_training(model, model_args,
                    dna_model_finetune=getattr(model_args, "dna_model_finetune", False),
                    dna_projection_finetune=getattr(model_args, "dna_projection_finetune", False))

    else:
        # 没有 SFT checkpoint，直接从预训练模型开始 GRPO 训练
        print(f"CALLING FOURTH PREP_FOR_TRAINING with dna_model_finetune: {getattr(model_args, 'dna_model_finetune', False)}")
        _ = _prep_for_training(model, model_args,
            dna_model_finetune=getattr(model_args, "dna_model_finetune", False),
            dna_projection_finetune=getattr(model_args, "dna_projection_finetune", False))

    # 可选：加载完整的训练 checkpoint（包含 optimizer state 等，用于恢复训练）
    if script_args.full_ckpt is not None:
        print(f"Loading full checkpoint from {script_args.full_ckpt}")
        checkpoint_path = os.path.join(script_args.full_ckpt, "pytorch_model.bin")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            print(f"Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        else:
            print(f"Checkpoint file not found at {checkpoint_path}")

    # 将模型移到 GPU
    model = model.to(training_args.device)

    # ── 步骤3：获取 DNA 模块和 reward 函数 ──
    vlm_module_cls = get_vlm_module(model_args.text_model_name)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    print("reward_funcs:", reward_funcs)
    print(f"Tokenizer loaded successfully. Vocab size: {len(model.text_tokenizer)}")
    print(f"ID for '<|dna_pad|>' is: {model.text_tokenizer.convert_tokens_to_ids('<|dna_pad|>')}")

    # ── 步骤4：加载数据集 ──
    dataset = get_kegg_questions(truncate_dna_per_side=model_args.truncate_dna_per_side)

    # 自定义保存回调
    custom_save_callback = SaveWithPyTorchCallback()
    print("model.text_model:", model.text_model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_trainable:,} / {total_params:,} ({100 * total_trainable / total_params:.2f}%)")

    # ── 步骤5：创建 GRPO Trainer 并开始训练 ──
    print("training_args:", training_args)
    trainer = DNALLMGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        dna_module=vlm_module_cls(),
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'] if training_args.eval_strategy != "no" else None,
        peft_config=None,
        callbacks=[custom_save_callback],
        processing_class=model.processor,
    )

    training_args.save_safetensors = False

    # 打印可训练参数详情
    print("="*50)
    print("Verifying Trainable Parameters...")
    print("="*50)
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [TRAINABLE]: {name} | Size: {param.shape} | Device: {param.device}")
            total_trainable_params += param.numel()
    print(f"\nTotal number of trainable parameters: {total_trainable_params:,}")
    print("="*50)

    # ── 处理 checkpoint 恢复 ──
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if resume_from_checkpoint == "True" or resume_from_checkpoint == "true":
        # 自动检测最新的 checkpoint
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = str(max(checkpoints, key=os.path.getmtime))
            print(f"Auto-resuming from latest checkpoint: {resume_from_checkpoint}")
        else:
            print("No checkpoints found to resume from. Starting fresh training.")
            resume_from_checkpoint = None
    elif resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # 开始训练！
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    # 环境变量设置
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # 禁用 HuggingFace datasets 的多进程（避免多 rank 冲突）
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    # 设置 wandb 项目名
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo")

    # 用 TRL 的 TrlParser 解析命令行参数
    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # 使用 PyTorch 原生保存（非 safetensors）
    training_args.save_safetensors = False
    # 优先使用环境变量中的 vLLM 服务器地址
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
