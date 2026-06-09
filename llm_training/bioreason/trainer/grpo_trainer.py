"""
DNALLMGRPOTrainer —— 基于 GRPO 算法的 DNA-LLM 强化学习训练器。
=============================================================================
GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的强化学习微调方法。

核心思想（对比标准 PPO）：
  标准 PPO: 需要 4 个模型 —— Policy、Reference、Reward、Value（Critic）
  GRPO:    只需要 2 个模型 —— Policy、Reference（Reward 用规则函数代替，不需要 Value）

GRPO 训练流程（每个 step）：
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Step 1: 生成阶段                                                     │
  │   对同一个 prompt，用当前 policy 生成 G 个不同的回复（组）             │
  │   使用 vLLM 高效生成（colocate 模式：vLLM 和训练共享 GPU）            │
  │                                                                     │
  │ Step 2: 评分阶段                                                     │
  │   用 reward 函数给每个回复打分（正确性、格式、简洁性等）               │
  │                                                                     │
  │ Step 3: 优势计算                                                     │
  │   advantage_i = reward_i - mean(rewards_in_group)                    │
  │   组内归一化：好的回复有正 advantage，差的回复有负 advantage           │
  │                                                                     │
  │ Step 4: 策略更新                                                     │
  │   用 PPO 风格的 clipped objective 更新 policy                        │
  │   loss = -min(ratio * advantage, clip(ratio) * advantage) + beta * KL│
  │                                                                     │
  │ Step 5: vLLM 权重同步（每 generation step）                           │
  │   将更新后的训练模型权重同步到 vLLM 引擎                               │
  └─────────────────────────────────────────────────────────────────────┘

关键设计：
  1. vLLM Colocation: 	vLLM 引擎和训练模型共享 GPU，避免数据搬运
  2. 批次缓存:         	生成的回复被缓存，多个 accumulation step 复用
  3. DNA 嵌入注入:     	生成时通过 get_prompt_embeddings 注入 DNA 信息
  4. 左填充:           	所有序列左填充（vLLM 要求），attention_mask 正确标记
  5. 重要性采样:       	vLLM 采样分布与训练模型分布可能不同，用 IS 修正

与 SFT Trainer 的关键区别：
  - SFT: 直接优化正确答案的 log probability
  - GRPO: 通过 reward 信号探索更好的生成策略，组内比较提供相对信号
=============================================================================
"""

import inspect
import os
import random
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import transformers

from functools import partial
from accelerate import logging
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_peft_available, is_datasets_available

# Monkey-patch: TRL >= 0.12 需要 PyTorch >= 2.4 的 FSDPModule,
# 服务器 PyTorch 较旧且使用 DeepSpeed (非 FSDP), 创建 dummy 绕过导入检查.
import torch.distributed.fsdp as _fsdp
if not hasattr(_fsdp, 'FSDPModule'):
    _fsdp.FSDPModule = type('FSDPModule', (), {})

# Monkey-patch: TRL >= 0.12 需要 transformers >= 4.46 的 is_trackio_available,
# 服务器 transformers 版本较旧, 补一个返回 False 的 dummy.
import transformers as _transformers
if not hasattr(_transformers, 'is_trackio_available'):
    _transformers.is_trackio_available = lambda: False

from trl.models import prepare_deepspeed, unwrap_model_for_generation, prepare_fsdp
from trl.models.utils import _ForwardRedirection
from trl.trainer.grpo_config import GRPOConfig
from trl.import_utils import is_liger_kernel_available, is_vllm_available

from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient

from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    RepeatSampler,
    entropy_from_logits,
    identity,
    pad,
    nanmax,
    nanmin,
    nanstd,
    selective_log_softmax,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

from accelerate.utils import is_peft_model, set_seed, gather_object

from torch.utils.data import Sampler

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

from bioreason.dataset.kegg import qwen_dna_collate_fn
from bioreason.dna_modules.dna_module import DNABaseModule
from bioreason.trainer import DNALLMGRPOConfig
from bioreason.utils.vllm_utils import should_update_and_canonicalize, fix_param_name_to_vllm, sync_fsdp1_params_to_vllm, sync_fsdp2_params_to_vllm

logger = logging.get_logger(__name__)

# Reward 函数类型：可以是字符串（模型ID）、预训练模型、或自定义函数
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], List[float]]]


class DNALLMGRPOTrainer(Trainer):
    """
    DNA-LLM 的 GRPO 训练器。

    继承自 HuggingFace Trainer，重写了核心训练逻辑：
    - get_train_dataloader():      使用更大的 generation batch
    - _prepare_inputs():           管理生成、评分、批次缓存的完整流程
    - _generate_and_score_completions(): 生成回复并计算 reward/advantage
    - compute_loss():              计算 GRPO 的 clipped loss
    - _move_model_to_vllm():       将训练模型权重同步到 vLLM 引擎

    vLLM 两种模式：
    1. colocate（本代码主要使用）: vLLM 和训练在同一 GPU 上，共享显存
       - 优点：不需要额外的 GPU/服务器，延迟低
       - 缺点：显存压力大（两个模型同时存在），需要 sleep/wake 切换
    2. server: vLLM 作为独立服务器运行
       - 优点：显存独立，更稳定
       - 缺点：需要额外的 GPU/服务器，网络延迟
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: DNALLMGRPOConfig = None,
        dna_module: DNABaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        **kwargs,
    ):
        # ── 基本参数初始化 ──
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.dna_module = dna_module

        # 模型初始化
        model_init_kwargs = args.model_init_kwargs or {}

        assert not isinstance(model, str), "model must NOT be a string in the current implementation"

        model_id = "Qwen/Qwen3-4B"

        # 获取模型 forward 方法的参数名（用于后续判断是否支持 logits_to_keep 等参数）
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # gradient checkpointing 和 use_cache 不能同时使用
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        # PEFT（LoRA）包装
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config, args)

        # ── 处理器（Tokenizer）初始化 ──
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)

        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.dna_token_id = model.dna_token_id

        # ── Reward 函数初始化 ──
        # 支持多种类型的 reward: 字符串（模型名）、HF模型、自定义函数
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                # 字符串：当作 HF 模型名加载
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward 权重：最终的 reward = sum(weight_i * reward_i)
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward 处理类（用于基于模型的 reward function 的 tokenization）
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # ── GRPO 训练参数 ──
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # 回复的最大 token 数 = |o_i|
        self.num_generations = args.num_generations              # 每个 prompt 生成的回复数量 (G)
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode  				# "colocate" 或 "server"
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards  # advantage 缩放策略: "group" / "batch" / "none"
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile

        if self.use_liger_loss and self.top_entropy_quantile < 1.0:
            raise NotImplementedError("Liger Kernels don't currently support masking token positions based on entropy.")
        if self.use_liger_loss and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling."
            )

        # ── 数据集 ──
        self.shuffle_dataset = args.shuffle_dataset

        # 不支持 IterableDataset（需要固定大小的数据集来管理 generation batch）
        if (isinstance(train_dataset, IterableDataset) or isinstance(eval_dataset, IterableDataset) or
            (isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values()))):
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # ── 生成配置 ──
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  				# GRPO 需要采样（探索），不能贪婪解码
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            pad_token_id=self.pad_token_id,
        )
        if hasattr(self.dna_module, "get_eos_token_id"):
            self.generation_config.eos_token_id = self.dna_module.get_eos_token_id(processing_class)

        # ── PPO clip 范围 ──
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # ── 多步更新 ──
        self.num_iterations = args.num_iterations  # 每组生成结果被重复使用的次数（μ）
        self._step = 0  				# 跟踪 forward + backward 的总步数
        self._buffered_inputs = None  	# 缓存生成的批次，跨 accumulation step 复用

        # 调用父类初始化
        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  	# data_collator 已在 get_train_dataloader 中处理
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # Trainer 默认会对 loss 做 gradient_accumulation_steps 的缩放
            # GRPO 的 loss 缩放逻辑不同（基于 completion token 总数），需要自己控制
            # 设置 compute_loss_func 为非 None 可以绕过 Trainer 的自动缩放
            compute_loss_func="non-None value to disable scaling",
        )

        # ── Reference Model（用于 KL 散度惩罚）──
        # 如果用了 PEFT，禁用 adapter 即可回退到 reference model，不需要单独复制
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif is_peft_model(model):
            self.ref_model = None
        else:
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)
            self.ref_model.config.use_cache = False

        # ── 初始化指标和日志 ──
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # 日志队列：固定大小，只保留最近一个 generation batch 的结果
        self._logs = {
            "dna_sequence": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # 设置不同的 seed，确保多 GPU 上生成的回复不同
        set_seed(args.seed, device_specific=True)

        # ── vLLM 初始化 ──
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True."
                )

            if self.vllm_mode == "server":
                # Server 模式：vLLM 作为独立服务运行
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Colocate 模式：vLLM 与训练共享 GPU（本代码主要使用）
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                # Tensor Parallel 分组
                if self.vllm_tensor_parallel_size > 1:
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                         for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)]
                    )

                # 设置分布式环境变量（vLLM 需要）
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)

                local_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                single_visible = bool(local_cvd) and ("," not in local_cvd) and (local_cvd.strip() not in ("", "-1"))
                os.environ["LOCAL_RANK"] = "0" if single_visible else str(self.accelerator.local_process_index)

                os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None

                # 创建 vLLM LLM 引擎
                # enable_prompt_embeds=True：关键！允许传入 embedding 而非 token IDs
                # 这样我们可以把 DNA embedding 注入到 prompt 中
                self.llm = LLM(
                    model=args.vllm_ckpt,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=10000,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    enable_prompt_embeds=True  # 关键：支持 prompt embeddings 输入
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)  # 开始时让 vLLM 休眠，节省显存
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            self._last_loaded_step = -1  # 跟踪上次同步权重到 vLLM 的 step
            self.accelerator.wait_for_everyone()  # 同步所有进程
        else:
            # 不使用 vLLM：用 HuggingFace 的 generate 方法
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
            }
            if args.use_transformers_paged:
                generation_kwargs["max_batch_tokens"] = 512
                generation_kwargs["num_blocks"] = 1024
                generation_kwargs["block_size"] = 128
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # 关闭 Trainer 的自动 loss 缩放（我们自己在 compute_loss 中处理）
        self.model_accepts_loss_kwargs = False

        # 添加模型标签
        self.model.text_model.add_model_tags(self._tag_names)
        self.current_gradient_accumulation_steps = int(getattr(self.args, "gradient_accumulation_steps", 1)) or 1

        # ── Reference Model 准备（分布式包装）──
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        # ── Reward Model 准备（分布式包装）──
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    # ═══════════════════════════════════════════════════════════════════════
    # DataLoader 和 Sampler
    # ═══════════════════════════════════════════════════════════════════════

    def get_train_dataloader(self):
        """
        重写的训练 DataLoader。

        关键改动（相比标准 Trainer）：
        batch_size = per_device_train_batch_size × steps_per_generation

        这意味着每个"批次"实际上是一个"generation batch"——包含 steps_per_generation 个
        标准批次的 prompt。生成回复一次，然后在多个 accumulation step 中复用。

        DataLoader 使用 qwen_dna_collate_fn 来：
        1. 渲染 chat template
        2. 处理 DNA 序列
        3. 构建 input_ids / attention_mask / labels
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = partial(
            qwen_dna_collate_fn,
            processor=self.processing_class,
            max_length_text=self.model.max_length_text,
            max_length_dna=self.model.max_length_dna,
            return_answer_in_batch=True,  # 返回原始答案供 reward 函数使用
        )
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # 关键改动：更大的 batch
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        """
        返回一个 RepeatSampler，确保：
        1. 每个 prompt 被重复 num_generations 次（用于生成 G 个回复）
        2. 相同的 prompt 被分配到不同的 GPU（用于组内 reward 归一化）

        采样示意图（num_generations=2, per_device_train_batch_size=3, steps_per_gen=4）:
                                         │   GPU 0  │   GPU 1  │
              global_step   step          │          │          │
           grad_accum=2  ▲  ▲  0    0     │ 0 0 1 1 2 2 │  生成第一批全部回复，使用第1个slice
                         ▼  │  0    1     │ 3 3 4 4 5 5 │  复用缓存，使用第2个slice
                            │  1    2     │ 6 6 7 7 8 8 │  复用缓存，使用第3个slice
           steps_per_gen=4  ▼  1    3     │ 9 9..        │  复用缓存，使用第4个slice
        """
        if dataset is None:
            dataset = self.train_dataset

        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,  	# 每个 prompt 重复 G 次
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """评估时也需要 RepeatSampler（每个 prompt 生成 G 个回复用于组内比较）"""
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _set_signature_columns_if_needed(self):
        """设置 signature columns 为 ["prompt"]，因为我们自己预处理数据"""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # ═══════════════════════════════════════════════════════════════════════
    # Log Probability 计算
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_logps_single_batch(self, model, input_ids, attention_mask, logits_to_keep, compute_entropy, **custom_multimodal_inputs):
        """
        计算单个 batch 的 per-token log probabilities。

        流程：
        1. 组织模型输入（包含 DNA 信息）
        2. 前向传播获取 logits
        3. 只保留最后 logits_to_keep 个位置的 logits（completion tokens）
        4. 除以 temperature（标准 RL 做法，控制策略的随机性）
        5. 计算 log_softmax 并取出实际 token 的 log probability

        logits_to_keep 的设计原因：
        - 我们只需要 completion tokens 的 logprob
        - prompt tokens 的 logprob 不需要，计算了也是浪费
        - 加 1 是因为最后一个 logit 对应"下一个"token，需要排除
        """
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'use_cache': False,
            **custom_multimodal_inputs
        }

        # 如果模型支持 logits_to_keep，可以减少计算量
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**model_inputs).logits  	# (B, L, V)
        logits = logits[:, :-1, :]  				# 排除最后一个 logit（对应下一个 token）
        logits = logits[:, -logits_to_keep:, :]  	# 只保留 completion tokens 的 logits
        # 除以 temperature：控制策略的熵
        # temperature 越高 → 分布越平滑 → 探索性越强
        logits = logits / self.temperature

        completion_ids = input_ids[:, -logits_to_keep:]
        logps = selective_log_softmax(logits, completion_ids)  # (B, logits_to_keep)

        entropies = None
        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)
        return logps, entropies

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, **custom_multimodal_inputs):
        """获取 per-token log probabilities（不计算 entropy）"""
        logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask,
            logits_to_keep=logits_to_keep, compute_entropy=False,
            **custom_multimodal_inputs
        )
        return logps

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep,
        compute_entropy, batch_size=None, **custom_multimodal_inputs
    ):
        """
        分批计算 per-token log probabilities（可选：同时计算 entropy）。

        为什么要分批（micro-batching）：
        - 训练模型的 forward 需要显存
        - 一次处理整个 generation batch 可能导致 OOM
        - 分成小批处理，用时间换空间

        DNA 特殊处理：
        - dna_tokenized 是按 DNA 序列编号的（不是按 batch 编号）
        - batch_idx_map 记录了每条 DNA 序列属于哪个 batch item
        - 切分子 batch 时需要重新映射 batch_idx_map
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size * 2

            # 处理 DNA 相关的输入（按序列编号而非 batch 编号）
            sliced_multimodal_inputs = {}
            for k, v in custom_multimodal_inputs.items():
                if k == 'dna_tokenized' and v is not None:
                    batch_idx_map = custom_multimodal_inputs.get('batch_idx_map', [])
                    if batch_idx_map:
                        dna_seq_indices = [i for i, batch_idx in enumerate(batch_idx_map) if start <= batch_idx < end]
                        sliced_multimodal_inputs[k] = {
                            'input_ids': v['input_ids'][dna_seq_indices] if len(dna_seq_indices) > 0 else v['input_ids'][:0],
                            'attention_mask': v['attention_mask'][dna_seq_indices] if len(dna_seq_indices) > 0 else v['attention_mask'][:0],
                        }
                    else:
                        sliced_multimodal_inputs[k] = v
                elif k == 'batch_idx_map' and v is not None:
                    # 重新编号：使索引从 0 开始（因为每个子 batch 是独立的）
                    sliced_map = [batch_idx - start for batch_idx in v if start <= batch_idx < end]
                    sliced_multimodal_inputs[k] = sliced_map
                else:
                    sliced_multimodal_inputs[k] = v[start:end] if isinstance(v, torch.Tensor) else v

            logps, entropies = self._compute_logps_single_batch(
                model, input_ids[start:end], attention_mask[start:end],
                logits_to_keep, compute_entropy, **sliced_multimodal_inputs
            )
            all_logps.append(logps)
            if compute_entropy:
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    # ═══════════════════════════════════════════════════════════════════════
    # vLLM 权重同步
    # ═══════════════════════════════════════════════════════════════════════

    @profiling_decorator
    def _move_model_to_vllm(self):
        """
        将训练模型的当前权重同步到 vLLM 引擎。

        为什么需要这一步：
        - vLLM 是一个独立的推理引擎，有自己的模型副本
        - 训练更新了模型权重后，vLLM 中的副本不会自动更新
        - 每次 generation 前需要手动同步，确保 vLLM 用最新的权重生成

        PEFT（LoRA）特殊处理：
        - 同步前需要 merge_adapter()：把 LoRA 权重合并到基础权重
        - 同步后需要 unmerge_adapter()：恢复 LoRA 状态，继续训练
        - 合并/取消合并都在 GatheredParameters 上下文中进行（DeepSpeed ZeRO-3 要求）

        支持的分布式策略：
        - DeepSpeed ZeRO Stage 3: 需要先 gather 参数
        - FSDP v1/v2: 使用专用的同步函数
        - DDP: 直接遍历参数
        """
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model.text_model):
            print("Using PEFT model: merging adapters before vLLM update.")
            # PEFT 模型：先合并 LoRA 适配器到基础权重
            with gather_if_zero3(list(self.model.text_model.parameters())):
                self.model.text_model.merge_adapter()
                print("Parameters merged for vLLM update.")

                if self.is_fsdp_enabled:
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        sync_fsdp1_params_to_vllm(self.accelerator, self.vllm_mode, self.vllm_client, self.llm, self.model.text_model)
                    elif fsdp_version == 2:
                        sync_fsdp2_params_to_vllm(self.llm, self.accelerator, self.vllm_mode, self.vllm_client, self.model.text_model)
                else:
                    # DeepSpeed ZeRO-3 + PEFT
                    for name, param in self.model.text_model.named_parameters():
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.text_model.prefix in name:
                            continue
                        if "original_module" in name:
                            continue
                        name = fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            name = should_update_and_canonicalize(name)
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

                # 恢复 LoRA 状态（unmerge）
                self.model.text_model.unmerge_adapter()
                print("Parameters unmerged after vLLM update.")
        else:
            # 非 PEFT：直接同步每个参数
            print("Non-PEFT model: updating vLLM with current parameters.")
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    sync_fsdp1_params_to_vllm(self.llm, self.accelerator, self.vllm_mode, self.vllm_client, self.model.text_model)
                elif fsdp_version == 2:
                    sync_fsdp2_params_to_vllm(self.llm, self.accelerator, self.vllm_mode, self.vllm_client, self.model.text_model)
            else:
                for name, param in self.model.text_model.named_parameters():
                    name = fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # 同步后重置 vLLM 的 prefix cache（权重变了，旧的 KV cache 无效）
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # 核心训练流程：_prepare_inputs
    # ═══════════════════════════════════════════════════════════════════════

    def _prepare_inputs(self, generation_batch: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        准备训练输入 —— GRPO 的核心编排方法。

        训练模式下的逻辑：
        ┌──────────────────────────────────────────────────────────────────┐
        │ 每隔 steps_per_generation × num_iterations 步：                     │
        │   1. 调用 _generate_and_score_completions() 生成回复并打分         │
        │   2. 打乱生成的回复（避免顺序偏差）                                 │
        │   3. 分成 steps_per_generation 个子批次                           │
        │   4. 缓存到 self._buffered_inputs                                │
        │                                                                  │
        │ 其他步：                                                           │
        │   直接返回缓存中的对应子批次（复用之前的生成结果）                   │
        └──────────────────────────────────────────────────────────────────┘

        这种设计的原因：
        - 生成回复很昂贵（vLLM 推理），不能每个 accumulation step 都生成
        - 同一组生成结果可以被多个 optimizer step 复用（num_iterations > 1）
        - DNA 相关字段需要特殊处理（shuffle 后 batch_idx_map 需要重新映射）
        """
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # 需要重新生成回复
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)

                # 提取 DNA 相关字段（在 shuffle 前取出）
                dna_tokenized = generation_batch.pop("dna_tokenized", None)
                batch_idx_map = generation_batch.pop("batch_idx_map", None)
                multimodal_inputs = generation_batch.pop("multimodal_inputs", None)

                # 随机打乱（避免模型学到生成顺序的偏差）
                batch_size = len(generation_batch["advantages"])
                permutation = list(range(batch_size))
                random.shuffle(permutation)

                for key, val in generation_batch.items():
                    if isinstance(val, torch.Tensor):
                        generation_batch[key] = val[permutation]
                    elif isinstance(val, list):
                        generation_batch[key] = [val[i] for i in permutation]

                # 更新 batch_idx_map：生成结果的顺序变了，DNA序列的映射也要跟着变
                if batch_idx_map is not None:
                    inverse_perm = [0] * batch_size
                    for new_idx, old_idx in enumerate(permutation):
                        inverse_perm[old_idx] = new_idx
                    batch_idx_map = [inverse_perm[idx] for idx in batch_idx_map]
                    if multimodal_inputs is not None:
                        multimodal_inputs["batch_idx_map"] = batch_idx_map

                # 将 generation batch 拆分为多个子批次
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)

                # 将 DNA 数据分配给每个子批次
                # 关键：DNA 数据按序列编号，需要正确分配到对应的子批次
                if dna_tokenized is not None or batch_idx_map is not None or multimodal_inputs is not None:
                    batch_idx_map_halved = torch.tensor(batch_idx_map[::2])  # 每两条 DNA 属于同一个 batch item
                    inv_map_partial = torch.argsort(batch_idx_map_halved)
                    inv_map = torch.stack((2*inv_map_partial, 2*inv_map_partial+1), dim=1).reshape(-1)
                    chunk_size = 2 * batch_size // self.args.steps_per_generation  # 2条DNA per item

                    dna_input_ids = dna_tokenized['input_ids'][inv_map]
                    dna_attention_mask = dna_tokenized['attention_mask'][inv_map]

                    for i, batch in enumerate(generation_batches):
                        batch["dna_tokenized"] = {
                            'input_ids': dna_input_ids[i * chunk_size:(i + 1) * chunk_size],
                            'attention_mask': dna_attention_mask[i * chunk_size:(i + 1) * chunk_size],
                        }
                        batch["batch_idx_map"] = torch.arange(0, chunk_size).tolist()
                        if multimodal_inputs is not None:
                            batch["multimodal_inputs"] = {
                                'dna_tokenized': batch["dna_tokenized"],
                                'batch_idx_map': batch["batch_idx_map"],
                            }

                # 缓存所有子批次（detach 防止跨 step 梯度累积）
                self._buffered_inputs = []
                for batch in generation_batches:
                    detached_batch = {}
                    for key, value in unsplit_pixel_values_by_grid(batch).items():
                        if isinstance(value, torch.Tensor):
                            detached_batch[key] = value.detach()
                        else:
                            detached_batch[key] = value
                    self._buffered_inputs.append(detached_batch)

            # 返回当前 accumulation step 对应的缓存子批次
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # 评估模式：不需要缓存，直接生成
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    # ═══════════════════════════════════════════════════════════════════════
    # Reward 计算
    # ═══════════════════════════════════════════════════════════════════════

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """
        计算所有 reward 函数的分值。

        Reward 函数类型：
        1. 神经网络模型（nn.Module）：用模型评分
        2. 规则函数（callable）：直接调用函数评分

        每个 reward 函数接收 prompts 和 completions，返回一个分数列表。

        Returns:
            rewards_per_func: (num_prompts, num_reward_funcs) 的 tensor
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # 准备 extra kwargs（排除 prompt/completion/completion_ids）
        keys = [key for key in inputs.keys() if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: inputs[key] for key in keys}
        reward_kwargs["trainer_state"] = self.state  # 允许根据训练进度动态调整 reward

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    # 模型类 reward：拼接 prompt+completion，用模型打分
                    texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True,
                        padding_side="right", add_special_tokens=True
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    # 规则类 reward：直接调用函数
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions,
                        completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # 如果某行所有 reward 都是 NaN，发出警告
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # 跨进程 gather reward（因为不同的 completion 分布在不同 GPU 上）
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    # ═══════════════════════════════════════════════════════════════════════
    # vLLM Colocate 生成
    # ═══════════════════════════════════════════════════════════════════════

    def _vllm_colocate(
        self, prompts_text, dna_tokenized, batch_idx_map,
        prompt_ids, prompt_mask, device
    ):
        """
        使用 colocated vLLM 引擎生成回复。

        这是本项目 GRPO 训练的核心创新之一：DNA embedding 注入到 vLLM 生成中。

        流程：
        1. （如果需要）唤醒 vLLM（sleep mode 下）
        2. （如果需要）同步训练模型权重到 vLLM
        3. 通过 model.get_prompt_embeddings() 将 DNA 嵌入到 prompt 中
           ── 这是关键：prompt_embeds 包含了 DNA 编码器提取的特征
        4. 用 vLLM 的 prompt_embeds 接口生成（而非普通 token ids）
        5. （如果需要）让 vLLM 休眠（释放显存给训练）

        Tensor Parallel 处理：
        - TP > 1 时，需要 all_gather 所有 rank 的 prompt
        - 所有 rank 都生成完整回复，然后各取所需的部分
        """
        if not self.use_vllm or self.vllm_mode != "colocate":
            return

        # Step 1: 唤醒 vLLM
        if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
            torch.cuda.empty_cache()
            self.llm.wake_up()

        # Step 2: 同步权重（如果训练步数已更新）
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Step 3: vLLM 采样参数
        generation_kwargs = {
            "n": 1,  # colocate 模式下每个 GPU 只生成 1 个回复 per prompt
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1 if self.top_k is None else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "max_tokens": self.max_completion_length,
            "logprobs": 0,  # 只返回生成 token 的 logprob
        }
        if self.args.generation_kwargs is not None:
            generation_kwargs.update(self.args.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Tensor Parallel: gather 所有 rank 的数据
        if self.vllm_tensor_parallel_size > 1:
            orig_size = len(prompts_text)
            gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
            gathered_dna_tokenized = [None for _ in range(self.vllm_tensor_parallel_size)]
            gathered_batch_idx_map = [None for _ in range(self.vllm_tensor_parallel_size)]
            gathered_prompt_ids = [None for _ in range(self.vllm_tensor_parallel_size)]
            gathered_prompt_mask = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
            torch.distributed.all_gather_object(gathered_dna_tokenized, dna_tokenized, group=self.tp_group)
            torch.distributed.all_gather_object(gathered_batch_idx_map, batch_idx_map, group=self.tp_group)
            torch.distributed.all_gather_object(gathered_prompt_ids, prompt_ids, group=self.tp_group)
            torch.distributed.all_gather_object(gathered_prompt_mask, prompt_mask, group=self.tp_group)
            all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            all_dna_tokenized = [p for sublist in gathered_dna_tokenized for p in sublist]
            all_batch_idx_map = [p for sublist in gathered_batch_idx_map for p in sublist]
            all_prompt_ids = [p for sublist in gathered_prompt_ids for p in sublist]
            all_prompt_mask = [p for sublist in gathered_prompt_mask for p in sublist]
        else:
            all_prompts_text = prompts_text
            all_dna_tokenized = dna_tokenized
            all_batch_idx_map = batch_idx_map
            all_prompt_ids = prompt_ids
            all_prompt_mask = prompt_mask

        # Step 4: 获取 prompt embeddings（包含 DNA 信息）
        # 这是关键步骤：DNA 编码器提取特征 → projection → 替换 <|dna_pad|>
        self.model.text_model.eval()
        with torch.inference_mode():
            prompt_embeds, attention_mask = self.model.get_prompt_embeddings(
                input_ids=all_prompt_ids.to(self.model.device),
                attention_mask=all_prompt_mask,
                dna_tokenized=all_dna_tokenized,
                batch_idx_map=all_batch_idx_map
            )
        text_embeddings = [prompt_embeds[i] for i in range(prompt_embeds.shape[0])]
        # 去掉 padding 部分（attention_mask=0 的位置）
        text_embeddings = [emb[attention_mask[i].bool()] for i, emb in enumerate(text_embeddings)]
        self.model.text_model.train()

        # Step 5: vLLM 生成（使用 prompt_embeds 而非 token ids）
        with profiling_context(self, "vLLM.generate"):
            all_outputs = self.llm.generate(
                [{"prompt_embeds": embed} for embed in text_embeddings],
                sampling_params=sampling_params, use_tqdm=True
            )

        # 提取生成的 token ids 和 logprobs
        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        all_logprobs = [
            [next(iter(lp.values())).logprob for lp in output.logprobs]
            for outputs in all_outputs
            for output in outputs.outputs
        ]

        # 边界检查：确保 completion 数量正确（处理 OOM 截断的情况）
        expected_local = len(all_prompts_text)
        got_local = len(completion_ids)
        if got_local != expected_local:
            print(f"[vLLM colocate] Warning: expected {expected_local} completions, got {got_local}.")
            if got_local < expected_local:
                missing_local = expected_local - got_local
                placeholder_ids = [self.eos_token_id]
                placeholder_logprobs = [0.0]
                for _ in range(missing_local):
                    completion_ids.append(placeholder_ids)
                    all_logprobs.append(placeholder_logprobs)
            else:
                completion_ids = completion_ids[:expected_local]
                all_logprobs = all_logprobs[:expected_local]

        # Tensor Parallel: 各取所需的部分
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
            completion_ids = completion_ids[tp_slice]
            all_logprobs = all_logprobs[tp_slice]

        # 让 vLLM 休眠，释放显存给训练
        if self.args.vllm_enable_sleep_mode:
            self.llm.sleep(level=1)

        torch.cuda.empty_cache()

        # 将 completion 转为 tensor 并做 padding
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        sampling_per_token_logps = [
            torch.tensor(logprobs, device=device, dtype=torch.float32) for logprobs in all_logprobs
        ]
        sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)

        return completion_ids, prompt_completion_ids, sampling_per_token_logps

    # ═══════════════════════════════════════════════════════════════════════
    # 生成和评分的主流程
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_and_score_completions(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        生成回复并计算 reward/advantage —— GRPO 的核心数据流。

        完整流程：
        1. 生成回复（vLLM 或 HuggingFace generate）
        2. 构建 completion_mask（EOS 之后的部分 mask 掉）
        3. 计算 old_per_token_logps（当前策略在 completion 上的 log prob）
        4. （可选）计算 ref_per_token_logps（reference model 的 log prob）
        5. 计算 rewards（多个 reward 函数）
        6. 组内归一化计算 advantage（reward - mean_group_reward）
        7. （可选）重要性采样修正（vLLM 采样分布 vs 训练模型分布）
        8. 日志记录（长度、reward、clip ratio 等）

        Returns:
            包含 prompt_ids, completion_ids, advantages, multimodal_inputs 等字段的字典
        """
        device = self.accelerator.device

        prompts_text = inputs["prompt"]
        original_prompts = inputs.get("original_prompts", prompts_text)  # 用于 reward 函数

        dna_tokenized = inputs.get("dna_tokenized")
        batch_idx_map = inputs.get("batch_idx_map")

        prompt_ids, prompt_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

        # ── 步骤1：生成回复 ──
        if self.use_vllm:
            # vLLM 模式：高效批量生成（colocate）
            completion_ids, prompt_completion_ids, sampling_per_token_logps = self._vllm_colocate(
                prompts_text, dna_tokenized, batch_idx_map,
                prompt_ids, prompt_mask, device
            )
        else:
            # HuggingFace generate 模式
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                kwargs = {k: v for k, v in inputs.items() if k not in self.dna_module.get_non_generate_params()}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        kwargs[k] = v.to(device)
                start = time.time()
                prompt_completion_ids = unwrapped_model.generate(
                    **kwargs, generation_config=self.generation_config, disable_compile=True
                )
                end = time.time()
                print(f"Generation time: {end - start:.9f} seconds")
            prompt_length = prompt_ids.size(1)
            if not self.dna_module.is_embeds_input():
                prompt_completion_ids = prompt_completion_ids
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                completion_ids = prompt_completion_ids
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # ── 步骤2：构建 completion_mask ──
        # EOS token 之后的所有 token 都应该被忽略（模型在 EOS 之后生成的内容无效）
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 转换为 list of lists（给 reward 函数用）
        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)

        # 可选：截断的 completion（没有 EOS 的）全部 mask 掉
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # 拼接 prompt 和 completion 的 attention_mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        logits_to_keep = completion_ids.size(1)
        mode = "train" if self.model.training else "eval"
        batch_size = (self.args.per_device_train_batch_size * self.args.steps_per_generation) if mode == "train" else self.args.per_device_eval_batch_size

        # ── 步骤3+4：计算 log probabilities ──
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations

            # 旧策略的 log prob（用于 PPO ratio = π_new / π_old）
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    model=self.model, input_ids=prompt_completion_ids,
                    attention_mask=attention_mask, compute_entropy=False,
                    dna_tokenized=dna_tokenized, batch_idx_map=batch_idx_map,
                    logits_to_keep=logits_to_keep, batch_size=batch_size,
                )
            else:
                old_per_token_logps = None

            # 重要性采样修正（vLLM 采样分布 ≠ 训练模型分布）
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Reference model 的 log prob（用于 KL 散度惩罚）
            if self.beta != 0.0:
                if self.ref_model is None:
                    cm = self.accelerator.unwrap_model(self.model.text_model).disable_adapter()
                else:
                    cm = nullcontext()

                with cm:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        model=self.model if self.ref_model is None else self.ref_model,
                        input_ids=prompt_completion_ids, attention_mask=attention_mask,
                        compute_entropy=False, dna_tokenized=dna_tokenized,
                        batch_idx_map=batch_idx_map, logits_to_keep=logits_to_keep,
                        batch_size=batch_size,
                    )
            else:
                ref_per_token_logps = None

        # 解码生成的文本
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]

        # ── 步骤5：计算 rewards ──
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        # 加权求和：total_reward = Σ(weight_i × reward_i)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        print("rewards:", rewards)

        # ── 步骤6：组内归一化计算 advantage ──
        # GRPO 的核心：同一 prompt 的 G 个回复之间比较
        # advantage_i = reward_i - mean(rewards_in_group)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        print("advantages:", advantages)

        # Advantage 缩放（按组内标准差，使 training 更稳定）
        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(f"Invalid value for scale_rewards: {self.scale_rewards}.")

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # 防止全零 advantage（会导致梯度消失）
        if torch.all(advantages == 0):
            advantages = advantages + 1e-6

        # ── 步骤7：只保留当前进程的部分 ──
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # ── 步骤8：日志记录 ──
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Completion 长度统计
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # EOS 终止率统计
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # 每个 reward 函数的统计
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # 日志队列
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # 重要性采样统计
        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item())
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item())

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_isr = torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            mean_isr = torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            max_isr = torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(nanmin(self.accelerator.gather(min_isr)).item())
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(self.accelerator.gather(mean_isr).nanmean().item())
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(nanmax(self.accelerator.gather(max_isr)).item())

        # ── 返回结果 ──
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "dna_tokenized": dna_tokenized,
            "batch_idx_map": batch_idx_map,
            "advantages": advantages,
            "multimodal_inputs": {
                "dna_tokenized": dna_tokenized,
                "batch_idx_map": batch_idx_map,
            },
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output

    # ═══════════════════════════════════════════════════════════════════════
    # Loss 计算
    # ═══════════════════════════════════════════════════════════════════════

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算 GRPO loss。

        GRPO Loss 公式：
        ┌─────────────────────────────────────────────────────────────────┐
        │ ratio = exp(π_new - π_old)                                      │
        │ L_clip = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)            │
        │ L_kl = exp(r_ref - r_new) - (r_ref - r_new) - 1  (KL散度估计)   │
        │ L_total = L_clip + β * L_kl                                     │
        │                                                                 │
        │ 其中 A = advantage = reward - mean(group_rewards)               │
        └─────────────────────────────────────────────────────────────────┘

        PPO-style clipping 的作用：
        - 防止策略更新太激进（ratio 被限制在 [1-ε, 1+ε]）
        - min(ratio*A, clip(ratio)*A)：取悲观估计，防止"好"的更新被过度放大
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]

        # 拼接完整序列
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # 当前策略的 log prob
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, **multimodal_inputs)

        # 旧策略的 log prob（如果 num_iterations > 1）或当前策略的 detach
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        mode = "train" if model.training else "eval"

        # ── PPO Clipped Loss ──
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)  		# ratio = π_new / π_old
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)  # clipped ratio
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)  	# 取悲观估计

        # ── KL 散度惩罚（可选）──
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            # KL 散度的近似估计（k3 估计器）
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # ── 最终 loss（只对 completion tokens 计算）──
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Clip ratio 统计
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    # ═══════════════════════════════════════════════════════════════════════
    # 日志
    # ═══════════════════════════════════════════════════════════════════════

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        日志记录：将所有累积的指标取平均并输出。

        指标结构：{"train": {"metric_name": [values]}, "eval": {...}}
        输出格式：{"train/metric_name": mean_value, ...}
        """
        metrics = {}
        for mode, mode_metrics in self._metrics.items():
            for key, val in mode_metrics.items():
                if len(val) > 0:
                    metrics[f"{mode}/{key}"] = sum(val) / len(val)
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        # 清空累积的指标
        for mode_metrics in self._metrics.values():
            mode_metrics.clear()
