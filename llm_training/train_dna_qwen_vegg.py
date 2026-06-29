"""
================================================================================
DNA-LLM 多模态微调训练脚本
================================================================================

整体架构:
  CSV 数据 → format (构建 prompt dict) → DataLoader → collate_fn (渲染+分词+labels)
  → DNALLMModel.forward (DNA embedding + LLM) → loss → 反向传播

模型结构:
  输入有两个分支:
    1. 文本分支: Qwen3 (4B/1.7B) 作为主 LLM，加 LoRA 微调
    2. DNA 分支: Evo2 / NucleotideTransformer 作为 DNA encoder，冻结
  两个分支通过 dna_projection (Linear) 连接: DNA embedding → 投影到 LLM 的 hidden_size

训练数据流 (这是理解 Ground truth 的关键):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. CSV 加载                                                        │
  │    load_dataset("csv", data_files="BioReason_protein_LLM_Ready.csv")│
  │                                                                     │
  │ 2. format_variant_effect_for_dna_llm()  ← variant_effect.py       │
  │    构建 prompt dict:                                                │
  │    {                                                                │
  │      "prompt": [                                                    │
  │        {"role": "user", "content": [dna_placeholder, question]},    │
  │        {"role": "assistant",                                        │
  │         "reasoning_content": "The variant alters...",  ← 推理文字   │
  │         "content": "Answer: GOF"}                     ← 最终答案   │
  │      ],                                                             │
  │      "dna_sequences": [ref_seq, variant_seq],                       │
  │      "answer": "Pathogenic; Gain-of-Function (GOF)"                 │
  │    }                                                                │
  │                                                                     │
  │ 3. qwen_dna_collate_fn()  ← kegg.py                                │
  │    a) prepare_prompt(): chat template 渲染 prompt dict → 文本字符串 │
  │       渲染结果:                                                     │
  │       "<|im_start|>user\n...<|im_end|>\n                            │
  │        <|im_start|>assistant\n                                      │
  │        <think>\n{reasoning_content}\n</think>\n\n                   │
  │        {content}\n<|im_end|>\n"                                     │
  │    b) processor(tokenizer): 将文本字符串分词 → input_ids             │
  │    c) 构建 labels:                                                  │
  │       - 全部初始化为 -100 (忽略)                                     │
  │       - 找到 <|im_start|>assistant\n 的位置                         │
  │       - 从该位置到 <|im_end|> 的 token 复制为真实 label              │
  │       - 即: 只有 assistant 回复部分参与 loss 计算                    │
  │                                                                     │
  │ 4. _step() 中 Ground truth 的由来:                                  │
  │    ground_truth = tokenizer.decode(                                 │
  │        input_ids[labels != -100]   ← 这就是训练要预测的 token       │
  │    )                                                                 │
  │    → 如果 ground truth 的 <think> 里是 "Answer: GOF" 而非推理文字,  │
  │      说明步骤2的 reasoning_content 还是旧格式, 新代码没生效          │
  └─────────────────────────────────────────────────────────────────────┘

关键文件:
  - train_dna_qwen_vegg.py  (本文件): 训练主逻辑, LightningModule, DataLoader
  - bioreason/dataset/kegg.py:          qwen_dna_collate_fn (labels 构建)
  - bioreason/dataset/variant_effect.py: format 函数 (构建 prompt dict)
  - bioreason/models/dna_llm.py:        DNALLMModel (forward + generate)
  - bioreason/models/dl/chat_template_dl.py: CHAT_TEMPLATE (Jinja2 模板)
  - bioreason/models/dl/processing_dl.py:    DLProcessor (文本+DNA双分词)
================================================================================
"""

import csv
import gc
import os
import random
import re
import time
import traceback
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from typing import *

import torch
import wandb
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from bioreason.dataset.kegg import get_format_kegg_function, qwen_dna_collate_fn
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.variant_effect import (
    clean_variant_effect_example,
    clean_variant_effect_non_snv_example,
    get_format_variant_effect_function,
)
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.models.dna_llm import DNALLMModel, get_target_modules

from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
register_evo2_tokenizer()

# 多进程数据加载使用文件系统共享策略 (避免 CUDA tensor 共享问题)
torch.multiprocessing.set_sharing_strategy("file_system")
# 禁止 tokenizer 内部并行 (避免与 DataLoader 多进程冲突)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _make_collate_fn(processor, max_length_text, max_length_dna,
                     return_answer_in_batch, truncate_for_generation):
    """
    构建 collate_fn 的工厂函数。

    collate_fn 是 DataLoader 的核心回调，每个 batch 都会调用它。
    它将原始样本列表 (dict 格式) 转换为模型可以消费的 tensor batch。

    关键参数:
      truncate_for_generation:
        - False (train/val): 保留完整序列 (含 assistant 回复), labels 才有效
        - True  (test):     截断到 "<|im_start|>assistant\n" 为止,
                             让模型自己生成后面的内容
      return_answer_in_batch:
        - True:  batch 中额外返回 "answer" 字段 (test 评估用)
        - False: 不需要答案文本
    """
    return partial(
        qwen_dna_collate_fn,
        processor=processor,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
        return_answer_in_batch=return_answer_in_batch,
        truncate_for_generation=truncate_for_generation,
    )


def _make_processor(model):
    """
    构建 DLProcessor (双模态处理器)。

    DLProcessor 同时持有:
      - tokenizer:     Qwen3 的文本分词器
      - dna_tokenizer: Evo2/NT 的 DNA 分词器

    它的 __call__ 方法:
      1. 用 dna_tokenizer 分词所有 DNA 序列
      2. 将文本中的 <|dna_pad|> 占位符替换为 DNA token 数量对应的 pad token
      3. 用 tokenizer 分词文本
      4. 返回 {input_ids, attention_mask, dna_tokenized, batch_idx_map}
    """
    return DLProcessor(
        tokenizer=model.text_tokenizer,
        dna_tokenizer=model.dna_tokenizer,
    )


class DNALLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning Module — 封装训练循环、验证、测试的完整生命周期。

    Lightning 自动管理:
      - 训练循环 (training_step → backward → optimizer.step)
      - 验证循环 (validation_step, 自动开启 torch.no_grad + model.eval)
      - 测试循环 (test_step + on_test_epoch_end)
      - 分布式训练 (DDP / DeepSpeed)
      - checkpoint 保存和恢复
      - 日志记录 (wandb / tensorboard)

    你只需要实现:
      - __init__:       构建模型
      - _step:          单步前向 (train/val/test 共用)
      - *_dataloader:   提供数据
      - configure_optimizers: 优化器+调度器
      - on_test_epoch_end: 测试评估逻辑
    """

    def __init__(self, hparams):
        """
        初始化: 保存超参数, 构建模型, 设置 LoRA。

        hparams 来自 shell 脚本传参 (run_vegg_path_evo2_4B.sh)。
        """
        super().__init__()
        # save_hyperparameters 会把所有 hparams 存到 self.hparams 中,
        # 并在 checkpoint 中持久化, resume 时可以恢复
        self.save_hyperparameters(hparams)

        # ── 从 hparams 解包到便捷属性 ──
        self.text_model_name = self.hparams.text_model_name
        self.dna_model_name = self.hparams.dna_model_name
        self.cache_dir = self.hparams.cache_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.text_model_finetune = self.hparams.text_model_finetune
        self.dna_model_finetune = self.hparams.dna_model_finetune
        self.lora_rank = self.hparams.lora_rank
        self.lora_alpha = self.hparams.lora_alpha
        self.lora_dropout = self.hparams.lora_dropout
        self.max_length_dna = self.hparams.max_length_dna
        self.max_length_text = self.hparams.max_length_text
        self.dna_is_evo2 = self.hparams.dna_is_evo2
        self.dna_embedding_layer = self.hparams.dna_embedding_layer
        self.return_answer_in_batch = self.hparams.return_answer_in_batch
        self.merge_val_test_set = self.hparams.merge_val_test_set
        self.dataset_type = self.hparams.dataset_type

        # ── 构建双模态模型 ──
        # DNALLMModel 内部:
        #   1. 加载 Qwen3 (text_model) + 它的 tokenizer
        #   2. 加载 Evo2 / NT (dna_model) + 它的 tokenizer
        #   3. 创建 dna_projection (Linear: dna_hidden → text_hidden)
        self.model = DNALLMModel(
            text_model_name=self.text_model_name,
            dna_model_name=self.dna_model_name,
            cache_dir=self.cache_dir,
            max_length_dna=self.max_length_dna,
            max_length_text=self.max_length_text,
            text_model_finetune=self.text_model_finetune,
            dna_model_finetune=self.dna_model_finetune,
            dna_is_evo2=self.dna_is_evo2,
            dna_embedding_layer=self.dna_embedding_layer,
        )

        # 保存子模块引用, 方便访问
        self.text_model = self.model.text_model      # Qwen3 (因果LM)
        self.dna_model = self.model.dna_model         # Evo2 (DNA encoder)
        self.dna_projection = self.model.dna_projection  # 投影层
        self.tokenizer = self.model.text_tokenizer    # Qwen3 tokenizer

        # ── 冻结/微调参数 ──
        # 策略:
        #   - DNA encoder 默认冻结 (不参与训练, 只提取特征)
        #   - 文本 LLM 加 LoRA (只训练低秩适配器, 省显存)
        #   - dna_projection 全参数训练 (桥接两个模态)
        self.lora_config = self._prep_for_training()

    def _prep_for_training(self) -> LoraConfig:
        """
        配置哪些参数训练、哪些冻结, 并设置 LoRA。

        训练策略 (当前配置):
          dna_model_finetune = False  → DNA encoder 冻结
          text_model_finetune = True  → Qwen3 加 LoRA 微调
          dna_projection          → 全参数训练 (必须训练, 这是桥接层)

        LoRA 原理:
          对于每个 target 线性层 W (d×k), 不直接训练 W, 而是训练低秩分解:
            W' = W + ΔW = W + B·A
          其中 B (d×r), A (r×k), r 是 rank (如 32/64)
          参数量: d×r + r×k ≪ d×k

        这样 4B 模型只需要训练 ~几百MB 的 LoRA 参数 + projection 层,
        单卡 24GB/40GB 就能跑。
        """
        # ── DNA encoder: 冻结 ──
        if self.dna_model_finetune:
            pass  # 如果显式要求微调 DNA, 不做任何冻结
        else:
            if self.dna_is_evo2:
                # Evo2 模型结构是 self.dna_model.model (StripedHyena)
                for param in self.dna_model.model.parameters():
                    param.requires_grad = False
            else:
                # NT 模型: 直接冻结所有参数
                for param in self.dna_model.parameters():
                    param.requires_grad = False

        # ── 文本 LLM: 加 LoRA ──
        if self.text_model_finetune:
            target_modules = get_target_modules(self)
            lora_config = LoraConfig(
                r=self.lora_rank,              # LoRA 秩, 越大表达能力越强但参数越多
                lora_alpha=self.lora_alpha,     # LoRA 缩放因子, 实际学习率 = alpha/r
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,  # 对哪些层加 LoRA (q_proj, v_proj 等)
                init_lora_weights="gaussian",
                bias="none",                    # bias 不训练
                task_type="CAUSAL_LM",          # 因果语言模型任务
            )
            # prepare_model_for_kbit_training: 把模型准备好用于 LoRA 训练
            # (特别是启用 gradient checkpointing 等)
            self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_config)
        else:
            # 不微调文本模型: 冻结所有参数
            for param in self.text_model.parameters():
                param.requires_grad = False

        # ── Projection 层: 始终训练 ──
        # 这是 DNA embedding → LLM hidden 的桥接层, 必须训练
        for param in self.dna_projection.parameters():
            param.requires_grad = True

        return lora_config

    # ══════════════════════════════════════════════════════════════════════════════
    # 核心训练/验证/测试步骤
    # ══════════════════════════════════════════════════════════════════════════════

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        训练/验证/测试共享的单步前向逻辑。

        参数:
          batch:    collate_fn 返回的字典
          prefix:   "train" | "val" | "test"

        返回:
          loss tensor (scalar)

        流程:
          1. 从 batch 取出 input_ids, attention_mask, labels, dna_tokenized
          2. 送入 self.model.forward()
             → DNA 序列通过 Evo2 提取 embedding
             → embedding 通过 projection 映射到 LLM 空间
             → 替换 input_ids 中的 <|dna_pad|> 位置为 DNA embedding
             → Qwen3 前向传播, 计算 next-token cross-entropy loss
          3. 每隔 N 步, 用当前 batch 的第一个样本做一次推理生成 (用于监控)
          4. 记录 loss 和 lr 到 wandb
        """
        # test_step 直接走 on_test_epoch_end, 不在这里算 loss
        if prefix == "test":
            return {"loss": torch.tensor(0.0, device=self.device)}

        # ── 取出文本输入 ──
        input_ids = batch["input_ids"].to(self.device).long()
        attention_mask = batch["attention_mask"].to(self.device).long()
        # labels 形状和 input_ids 一样, 但 user 部分全是 -100 (忽略)
        labels = batch["labels"].to(self.device).long() if "labels" in batch else None

        # ── 取出 DNA 输入 ──
        dna_tokenized = batch.get("dna_tokenized")
        if dna_tokenized is not None:
            dna_tokenized = dna_tokenized.to(self.device)
            dna_tokenized["input_ids"] = dna_tokenized["input_ids"].long()
            if "attention_mask" in dna_tokenized:
                dna_tokenized["attention_mask"] = dna_tokenized["attention_mask"].long()

        # batch_idx_map: [0, 0, 1, 1, ...] 每条 DNA 序列属于 batch 中第几个样本
        batch_idx_map = batch.get("batch_idx_map")

        # ── 前向传播 ──
        # self.model.forward() 内部:
        #   1. Qwen3 embedding 层: input_ids → text_embeddings
        #   2. Evo2 编码 DNA: dna_tokenized → dna_embeddings
        #   3. dna_projection: dna_embeddings → 映射到 text_hidden_size
        #   4. 替换 text_embeddings 中 <|dna_pad|> 位置为 DNA embeddings
        #   5. Qwen3 forward(labels=labels) → 自动计算 cross-entropy loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
            labels=labels,
        )

        loss = outputs.loss

        # ── 定期采样生成 (仅用于监控训练进展, 不影响模型训练) ──
        # train: 每 3000 步; val: 每 300 个 batch
        if (prefix == "train" and (self.global_step % 3000 == 0)) or \
           (prefix == "val" and (batch_idx % 300 == 0)):
            try:
                example_idx = 0  # 用 batch 中第 0 个样本做演示
                print(f"\n=== Sample Generation (step {self.global_step} / "
                      f"{self.trainer.estimated_stepping_batches}) ===")

                # ── 找到 <|im_start|>assistant\n 标记的位置 ──
                # 推理时只把标记之前的内容作为输入, 让模型生成后面的部分
                assistant_start_marker = "<|im_start|>assistant\n"
                assistant_marker_tokens = self.tokenizer.encode(
                    assistant_start_marker, add_special_tokens=False)
                marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
                marker_len = len(assistant_marker_tokens)

                # 找到第一个非 padding token 的位置 (因为用了 left padding)
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                # 扫描找到 assistant 标记的位置
                matches = []
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        matches.append(pos)
                        break
                assistant_pos = matches[0] if matches else None

                if assistant_pos is not None:
                    # ── 截取输入: 从 start_idx 到 <|im_start|>assistant\n 之后 ──
                    # 这样模型只看到 user 的输入 + assistant 标记,
                    # 需要自己生成 <think>...</think> + answer
                    gen_input_ids = input_ids[
                        example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                    gen_attention_mask = attention_mask[
                        example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]

                    # ── 准备该样本对应的 DNA 数据 ──
                    example_dna_data = None
                    example_batch_map = None
                    if dna_tokenized is not None and batch_idx_map is not None:
                        example_indices = [i for i, idx in enumerate(batch_idx_map)
                                           if idx == example_idx]
                        if len(example_indices) > 0:
                            example_dna_data = BatchEncoding({
                                "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                                "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                            })
                            example_batch_map = [0] * len(example_indices)

                    # ── 推理生成 ──
                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            dna_tokenized=example_dna_data,
                            batch_idx_map=example_batch_map,
                            max_new_tokens=800,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    # ── 解码并打印 ──
                    # gen_input_ids: 模型看到的部分 (prompt)
                    user_input = self.tokenizer.decode(
                        gen_input_ids[0], skip_special_tokens=False).strip()
                    # generated: prompt + 模型生成的 tokens
                    generation = self.tokenizer.decode(
                        generated[0], skip_special_tokens=False).strip()

                    del generated, gen_input_ids, gen_attention_mask, example_dna_data, example_batch_map
                    gc.collect()

                    print(f"=====[Sample {prefix} {batch_idx}]=====")
                    print(f"=====[User input]=====\n{user_input}")
                    print(f"=====[Complete generation]=====\n{generation}")

                    # ── 打印 Ground truth (训练标签) ──
                    # 这是最关键的部分!!!
                    #
                    # labels != -100 的位置存储的是 assistant 回复的真实 token ID。
                    # 我们从 input_ids 中取出这些位置的 token, 解码后就是模型
                    # "应该"生成的内容 (即训练数据的 assistant 部分)。
                    #
                    # 如果这里解码出来 <think> 里是 "Answer: GOF" 而不是
                    # "The variant alters...", 说明训练数据的 reasoning_content
                    # 还是旧值, 也就是 format 函数没生效。
                    ground_truth = ""
                    if labels is not None:
                        # valid_label_pos: labels 不是 -100 的所有位置索引
                        valid_label_pos = (labels[example_idx] != -100).nonzero(as_tuple=True)[0]
                        if len(valid_label_pos) > 0:
                            if valid_label_pos[0] >= assistant_pos + marker_len:
                                # 用 tokenizer.decode 把这些位置的 token 解码成文本
                                ground_truth = self.tokenizer.decode(
                                    input_ids[example_idx, valid_label_pos],
                                    skip_special_tokens=False).strip()
                                print(f"=====[Ground truth]=====\n{ground_truth}")

                    # ── 记录到 wandb table ──
                    timestamp = time.time()
                    step_id = f"gen_{self.global_step}-{timestamp}"
                    wandb_logger = self.logger.experiment
                    wandb_logger.log({
                        step_id: wandb.Table(
                            columns=["timestamp", "prefix", "batch_idx",
                                     "user_input", "generation", "ground_truth"],
                            data=[[timestamp, prefix, batch_idx,
                                   user_input, generation, ground_truth]],
                        )
                    })

                    del user_input, generation, ground_truth
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print("No assistant marker found in the input sequence")

            except Exception as e:
                print(f"Error during sample generation: {str(e)}")
                traceback.print_exc()

        # ── 记录 loss 和学习率到 wandb ──
        if prefix != "test":
            current_lr = self.lr_schedulers().get_last_lr()[0]
        else:
            current_lr = 0

        # on_step: 记录每个 step 的值; on_epoch: epoch 结束时求平均
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        self.log(f"{prefix}_loss_epoch", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        if prefix != "test":
            self.log("lr", current_lr, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """每个训练 batch 调用一次。"""
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """每个验证 batch 调用一次, 自动关闭 dropout 和 batch norm。"""
        return self._step(batch, batch_idx, prefix="val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        每个测试 batch 调用一次。
        这里只占位, 真正的测试逻辑在 on_test_epoch_end 中。
        因为每条样本需要逐个生成, 不适合在 step 里做。
        """
        return self._step(batch, batch_idx, prefix="test")

    # ══════════════════════════════════════════════════════════════════════════════
    # 优化器和学习率调度
    # ══════════════════════════════════════════════════════════════════════════════

    def configure_optimizers(self):
        """
        配置 AdamW 优化器 + Cosine 学习率调度 (with linear warmup)。

        返回格式:
          [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        interval="step" 表示每个 optimizer step 更新一次 lr
        (而不是每个 epoch)。
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        # estimated_stepping_batches: 考虑了 grad_accum 后的总 optimizer step 数
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)  # 前 10% 步数线性升温
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ══════════════════════════════════════════════════════════════════════════════
    # 数据加载
    # ══════════════════════════════════════════════════════════════════════════════

    def _load_dataset_splits(self, split: str):
        """
        加载并格式化数据集, 按 split 返回对应子集。

        参数:
          split: "train" | "val" | "test"

        返回:
          (raw_dataset, labels_list)

        对于 variant_effect_coding:
          1. 从 CSV 加载 → HuggingFace Dataset
          2. clean_variant_effect_example: 清洗 answer 字段
          3. get_format_variant_effect_function: 构建 prompt dict
             (这里会调用 variant_effect.py 中的 format 函数,
              reasoning_content 就在这里被设置!)
          4. 80/10/10 分割: train / val / test
        """
        dtype = self.hparams.dataset_type

        if dtype == "kegg":
            # KEGG 数据集: HuggingFace hub 上的远程数据集
            dataset = load_dataset(self.hparams.kegg_data_dir_huggingface)
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))
            labels = []
            for sp, data in dataset.items():
                labels.extend(data["answer"])

            if split == "train":
                raw = dataset["train"]
            elif split == "val":
                raw = (concatenate_datasets([dataset["test"], dataset["val"]])
                       if self.hparams.merge_val_test_set else dataset["val"])
            else:  # test
                raw = (concatenate_datasets([dataset["test"], dataset["val"]])
                       if self.hparams.merge_val_test_set else dataset["test"])

        elif dtype == "variant_effect_coding":
            # ═══════════════════════════════════════════════════════════════════
            # 基因级分割 (Gene-Level Split)
            #
            # 核心原则: 同一基因的所有变异必须在同一个集合中 (train/val/test)
            # 这样模型无法靠"背基因名"作弊, 必须学习变异本身的特征
            #
            # 分层采样: 分别对纯GOF基因/纯LOF基因/混合基因做 8:1:1 分割
            # 确保每个集合的基因类型分布一致, 评估更公平
            # ═══════════════════════════════════════════════════════════════════
            stage = getattr(self.hparams, "stage", 1)
            if stage == 1:
                data_file = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage1_Binary_Reasoning.csv"
            else:
                data_file = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage2_GOF_LOF_Reasoning.csv"
            dataset = load_dataset("csv", data_files=data_file)
            raw_data = dataset["train"]

            # Step 1: 从 question 文本中提取基因名
            def _extract_gene(example):
                q = example.get("question", "")
                m = re.search(r"- Gene: (\S+)", q)
                return {"_gene": m.group(1) if m else ""}
            raw_data = raw_data.map(_extract_gene, load_from_cache_file=False)

            # Step 2: 统计每个基因的标签种类 (用于分层)
            gene_to_labels = defaultdict(set)
            gene_to_type = {}
            for example in raw_data:
                gene = example.get("_gene", "")
                if not gene:
                    continue
                ans = example.get("answer", "").strip().lower()
                gene_to_labels[gene].add(ans)
                gene_to_type[gene] = example.get("gene_type", "shared")

            # Step 3: 基因分类 (Stage 感知)
            #   Stage 1: Pathogenic vs Benign  → answer="pathogenic"/"benign"
            #   Stage 2: GOF vs LOF            → answer="gain-of-function"/"loss-of-function"
            if stage == 1:
                pure_path = []    # 只有 Pathogenic
                pure_benign = []  # 只有 Benign
                mixed_s1 = []     # 同时有 Pathogenic 和 Benign

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
                pure_gof = []
                pure_lof = []
                mixed_s2 = []
                other_s2 = []

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

            # Step 4: 分层 8:1:1 分割
            # 每个桶(pure/mixed)内独立分割, 保证各集合基因类型分布一致
            # 基因级隔离已解决泄露问题(同基因不会跨集合), 纯基因在test也可接受
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

            # Stage 2: lof_only 基因全部进 train, 不进 val/test
            if stage == 2:
                for gene, gtype in gene_to_type.items():
                    if gtype == "lof_only":
                        train_genes.add(gene)
                        val_genes.discard(gene)
                        test_genes.discard(gene)

            # Step 5: 按基因集合过滤 + 格式化
            if split == "train":
                gene_set = train_genes
            elif split == "val":
                gene_set = val_genes
            else:
                gene_set = test_genes

            raw = raw_data.filter(lambda x: x.get("_gene", "") in gene_set)
            raw = raw.map(get_format_variant_effect_function(self.hparams.model_type), load_from_cache_file=False)
            labels = [example["answer"] for example in raw]

        elif dtype == "variant_effect_non_snv":
            dataset = load_dataset(self.hparams.variant_effect_non_snv_data_dir_huggingface)
            dataset = dataset.map(clean_variant_effect_non_snv_example)
            cleaned = dataset.map(clean_variant_effect_example)
            labels = []
            for sp, data in cleaned.items():
                labels.extend(data["answer"])
            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
            train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            if split == "train":
                raw = train_val_split["train"]
                raw = raw.map(get_format_variant_effect_function(self.hparams.model_type))
            elif split == "val":
                raw = train_val_split["test"]
                raw = raw.map(get_format_variant_effect_function(self.hparams.model_type))
            else:  # test
                raw = dataset["test"]
                raw = raw.map(get_format_variant_effect_function(self.hparams.model_type))

        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        # 可选: 截断 DNA 序列 (每个方向保留指定长度)
        if self.hparams.truncate_dna_per_side:
            raw = raw.map(truncate_dna,
                          fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side})

        return raw, labels

    def train_dataloader(self) -> DataLoader:
        """
        训练数据加载器。

        truncate_for_generation=False (保留完整 assistant 回复),
        这样 labels 才包含 assistant 部分, loss 计算才有效。
        """
        raw, labels = self._load_dataset_splits("train")
        self.labels = sorted(list(set(labels)))  # 收集所有类别标签, 供测试评估用

        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=self.return_answer_in_batch,
            truncate_for_generation=False,  # ★ 训练时保留完整序列
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        """
        验证数据加载器。

        同训练: truncate_for_generation=False, 保留完整序列用于计算 val_loss。
        """
        raw, labels = self._load_dataset_splits("val")
        self.labels = sorted(list(set(labels)))

        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=self.return_answer_in_batch,
            truncate_for_generation=False,  # ★ 验证也保留完整序列
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        """
        测试数据加载器。

        关键区别:
          return_answer_in_batch=True   → batch 中返回 answer 字段, 用于评估
          truncate_for_generation=True  → 截断到 assistant 标记, 模型需要自己生成
        """
        raw, labels = self._load_dataset_splits("test")
        self.labels = sorted(list(set(labels)))

        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=True,       # ★ 测试需要答案文本
            truncate_for_generation=True,      # ★ 截断 prompt, 让模型生成
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    # ══════════════════════════════════════════════════════════════════════════════
    # 测试评估 (训练完成后自动调用)
    # ══════════════════════════════════════════════════════════════════════════════

    def on_test_epoch_end(self):
        """
        遍历整个测试集, 逐条生成并计算准确率。

        三个阶段评估:
          阶段一: Pathogenic vs Benign (致病性二分类)
          阶段二: GOF vs LOF vs Neutral (效应三分类)
          阶段三: GOF vs LOF (效应方向二分类, 剔除 Neutral)

        准确率判断: ground_truth 标签是否出现在生成的文本中 (忽略大小写)。
        只检查 </think> 之后的部分, 避免推理过程中碰巧提及标签。
        """
        wandb_logger = self.logger.experiment
        wandb_logger.log({"test_progress": 0.0, "status": "starting test generation"})

        self.model.eval()

        test_dataloader = self.test_dataloader()
        total_batches = len(test_dataloader)
        is_binary = len(self.labels) == 2  # 是否只有2个类别

        if is_binary:
            neg_label = self.labels[0]
            pos_label = self.labels[1]
            wandb_logger.log({"positive_label": pos_label, "negative_label": neg_label})
            print(f"Binary task — Positive: '{pos_label}', Negative: '{neg_label}'")
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0
        else:
            print(f"Multi-class task — {len(self.labels)} classes")

        total_examples = 0
        correct_count = 0
        both_count = 0
        truncated_count = 0
        processed_batches = 0
        generations = []  # 记录每条样本的生成结果

        for batch_idx, batch in enumerate(test_dataloader):
            wandb_logger.log({
                "test_progress": batch_idx / total_batches,
                "status": f"processing batch {batch_idx}/{total_batches}",
            })

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            answer = batch["answer"]  # 真实答案文本列表
            dna_tokenized = batch.get("dna_tokenized")
            if dna_tokenized is not None:
                dna_tokenized = dna_tokenized.to(self.device)
            batch_idx_map = batch.get("batch_idx_map")

            assistant_start_marker = "<|im_start|>assistant\n"
            assistant_marker_tokens = self.tokenizer.encode(
                assistant_start_marker, add_special_tokens=False)
            marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
            marker_len = len(assistant_marker_tokens)

            examples_in_batch = 0
            # 遍历 batch 中每个样本
            for example_idx in range(input_ids.size(0)):

                if total_examples % 10 == 0:
                    current_acc = correct_count / max(1, total_examples)
                    wandb_logger.log({"examples_processed": total_examples,
                                      "current_accuracy": current_acc})

                # 找非 padding 起始位置 (left padding)
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id
                           ).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                # 找 <|im_start|>assistant\n 标记位置
                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                if assistant_pos is None:
                    continue  # 没找到标记, 跳过这个样本

                # 截取: 只保留到 assistant 标记为止 (让模型生成后面部分)
                gen_input_ids = input_ids[
                    example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                gen_attention_mask = attention_mask[
                    example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]

                # 准备 DNA 数据
                example_dna_data = None
                example_batch_map = None
                if dna_tokenized is not None and batch_idx_map is not None:
                    example_indices = [i for i, idx in enumerate(batch_idx_map)
                                       if idx == example_idx]
                    if example_indices:
                        example_dna_data = BatchEncoding({
                            "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                            "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                        })
                        example_batch_map = [0] * len(example_indices)

                # 推理生成 (对齐原始BioReason: 采样解码)
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        dna_tokenized=example_dna_data,
                        batch_idx_map=example_batch_map,
                        max_new_tokens=800,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                user_input = self.tokenizer.decode(
                    gen_input_ids[0], skip_special_tokens=False).strip()
                generation = self.tokenizer.decode(
                    generated[0], skip_special_tokens=False).strip()

                ground_truth = answer[example_idx].strip()

                # ── 准确率判断 ──
                # 只检查 </think> 之后的部分，避免 think 块中提及标签造成虚高
                if ";" in ground_truth:
                    ground_truth = ground_truth.split(";")[0]

                # 截断保护: 没生成出 </think> 说明被 max_new_tokens 截断了
                if "</think>" in generation:
                    answer_part = generation.split("</think>")[-1]
                else:
                    # 被截断，未能输出答案，整条算无效
                    total_examples += 1
                    examples_in_batch += 1
                    truncated_count += 1
                    prediction_category = "TRUNCATED"
                    generations.append({
                        "batch_idx": batch_idx,
                        "example_idx": example_idx,
                        "user_input": user_input,
                        "generation": generation,
                        "ground_truth": ground_truth,
                        "correct": False,
                        "prediction_category": "TRUNCATED",
                    })
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                answer_lower = answer_part.lower()
                gt_lower = ground_truth.lower()

                # 检测模型是否同时输出了两个互斥标签 → 无效预测
                gof_keywords = ["gain-of-function", "gof"]
                lof_keywords = ["loss-of-function", "lof"]
                has_gof = any(kw in answer_lower for kw in gof_keywords)
                has_lof = any(kw in answer_lower for kw in lof_keywords)
                both_labels = has_gof and has_lof

                if both_labels:
                    # 同时输出 GOF 和 LOF → 模型在胡言乱语，不算正确
                    hit = False
                    prediction_category = "BOTH"
                    both_count += 1
                else:
                    hit = gt_lower in answer_lower
                    prediction_category = "correct" if hit else "wrong"

                total_examples += 1
                examples_in_batch += 1
                if hit:
                    correct_count += 1

                # ── 二分类指标 ──
                if is_binary:
                    is_pos = ground_truth.lower() == pos_label.lower()
                    is_neg = ground_truth.lower() == neg_label.lower()
                    if both_labels:
                        # 输出两个标签 → 无法判断，计入无效
                        pass  # 不更新 TP/FP/TN/FN
                    elif is_pos and hit:
                        true_positives += 1
                        prediction_category = "TP"
                    elif is_pos and not hit:
                        false_negatives += 1
                        prediction_category = "FN"
                    elif is_neg and hit:
                        true_negatives += 1
                        prediction_category = "TN"
                    elif is_neg and not hit:
                        false_positives += 1
                        prediction_category = "FP"

                # 保存本条结果
                generations.append({
                    "batch_idx": batch_idx,
                    "example_idx": example_idx,
                    "user_input": user_input,
                    "generation": generation,
                    "ground_truth": ground_truth,
                    "correct": hit,
                    "prediction_category": prediction_category,
                })

                torch.cuda.empty_cache()
                gc.collect()

            processed_batches += 1
            current_acc = correct_count / max(total_examples, 1)
            wandb_logger.log({
                "batches_processed": processed_batches,
                "examples_processed": total_examples,
                "current_accuracy": current_acc,
                "progress_percentage": (batch_idx + 1) / total_batches * 100,
            })

        # ── 计算最终指标 ────────────────────────────────────────────────────
        accuracy = correct_count / max(total_examples, 1)
        metrics = {
            "test_accuracy": accuracy,
            "total_examples_processed": total_examples,
            "correct_count": correct_count,
            "test_status": "completed",
        }

        if is_binary:
            # 二分类任务: 计算 precision, recall, F1
            precision = true_positives / max(true_positives + false_positives, 1)
            recall    = true_positives / max(true_positives + false_negatives, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-8)
            metrics.update({
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "both_labels_count": both_count,
                "truncated_count": truncated_count,
            })
            summary = (
                f"Test Results:\n"
                f"Total: {total_examples}  Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}\n"
                f"TP={true_positives} FP={false_positives} "
                f"TN={true_negatives} FN={false_negatives}\n"
                f"BOTH: {both_count}  TRUNCATED: {truncated_count}"
            )
        else:
            # 多分类任务: 三个阶段评估
            from sklearn.metrics import classification_report, accuracy_score

            y_true_3class = []   # 三分类真实标签
            y_pred_3class = []   # 三分类预测标签
            y_true_binary = []   # 二分类(Pathogenic vs Benign)真实标签
            y_pred_binary = []   # 二分类预测标签

            # 解析每条生成结果, 提取预测标签
            for g in generations:
                ans = g["generation"].lower()
                truth = g["ground_truth"]

                y_true_3class.append(truth)
                y_true_binary.append("Pathogenic" if "Pathogenic" in truth else "Benign")

                # 根据生成文本中的关键词判断预测类别
                # 注意: gof 必须在 lof 之前检查, 防止 "loss-of-function" 中
                # 的 "of" 被误判为 "gof" 的一部分 (实际不会, 但检查顺序仍重要)
                if "gof" in ans or "gain-of-function" in ans:
                    pred_3 = "Pathogenic; Gain-of-Function (GOF)"
                    pred_2 = "Pathogenic"
                elif "lof" in ans or "loss-of-function" in ans:
                    pred_3 = "Pathogenic; Loss-of-Function (LOF)"
                    pred_2 = "Pathogenic"
                elif "neutral" in ans or "benign" in ans:
                    pred_3 = "Benign; Neutral"
                    pred_2 = "Benign"
                else:
                    pred_3 = "Unknown/Invalid"
                    pred_2 = "Unknown/Invalid"

                y_pred_3class.append(pred_3)
                y_pred_binary.append(pred_2)

            # 阶段一: 致病性二分类 (Pathogenic vs Benign)
            bin_acc = accuracy_score(y_true_binary, y_pred_binary)
            bin_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)

            # 阶段二: 效应三分类 (GOF vs LOF vs Neutral)
            multi_acc = accuracy_score(y_true_3class, y_pred_3class)
            multi_report = classification_report(y_true_3class, y_pred_3class, zero_division=0)

            # 阶段三: GOF vs LOF 二分类 (剔除 Benign/Neutral, 只看致病方向)
            y_true_gof_lof = []
            y_pred_gof_lof = []
            for g in generations:
                truth = g["ground_truth"]
                truth_lower = truth.lower()
                is_gof_truth = "gof" in truth_lower or "gain-of-function" in truth_lower
                is_lof_truth = "lof" in truth_lower or "loss-of-function" in truth_lower
                if not (is_gof_truth or is_lof_truth):
                    continue  # 跳过 Benign/Neutral 样本

                y_true_gof_lof.append("GOF" if is_gof_truth else "LOF")

                ans = g["generation"].lower()
                if "</think>" in ans:
                    answer_part = ans.split("</think>")[-1]
                else:
                    answer_part = ans

                if "gof" in answer_part or "gain-of-function" in answer_part:
                    y_pred_gof_lof.append("GOF")
                elif "lof" in answer_part or "loss-of-function" in answer_part:
                    y_pred_gof_lof.append("LOF")
                else:
                    y_pred_gof_lof.append("Unknown")

            gof_lof_acc = accuracy_score(y_true_gof_lof, y_pred_gof_lof) if y_true_gof_lof else 0.0
            gof_lof_report = classification_report(
                y_true_gof_lof, y_pred_gof_lof, zero_division=0
            ) if y_true_gof_lof else "No GOF/LOF samples found."

            metrics["gof_lof_binary_accuracy"] = gof_lof_acc
            metrics["gof_lof_sample_count"] = len(y_true_gof_lof)

            summary = (
                f"\n=======================================================\n"
                f"🏆 顶会论文对标报告 (Two-Stage Evaluation)\n"
                f"=======================================================\n"
                f"▶ 阶段一：致病性二分类准确率 (Pathogenic vs Benign)\n"
                f"  Binary Accuracy: {bin_acc:.4f}\n"
                f"{bin_report}\n"
                f"-------------------------------------------------------\n"
                f"▶ 阶段二：效应三分类全景表现 (GOF vs LOF vs Neutral)\n"
                f"  Multi-class Accuracy: {multi_acc:.4f}\n"
                f"{multi_report}\n"
                f"-------------------------------------------------------\n"
                f"▶ 阶段三：效应方向二分类 (GOF vs LOF, {len(y_true_gof_lof)} samples)\n"
                f"  GOF/LOF Binary Accuracy: {gof_lof_acc:.4f}\n"
                f"{gof_lof_report}\n"
                f"=======================================================\n"
            )

        print(summary)
        wandb_logger.log({**metrics, "test_summary": summary})

        # ── 保存生成结果到 wandb table ──
        if generations:
            columns = list(generations[0].keys())
            wandb_logger.log({
                f"test_generations_{time.strftime('%Y%m%d-%H%M%S')}": wandb.Table(
                    columns=columns,
                    data=[[g.get(c, "") for c in columns] for g in generations],
                )
            })

        # ── 保存生成结果到 CSV ──
        model_name = self.hparams.text_model_name.split("/")[-1]
        csv_dir = (self.hparams.ckpt_path if self.hparams.ckpt_path
                   else self.hparams.checkpoint_dir)
        csv_path = os.path.join(
            csv_dir,
            f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv"
        )
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=generations[0].keys())
                writer.writeheader()
                writer.writerows(generations)
            wandb_logger.log({"csv_saved": True, "csv_path": csv_path})
        except Exception as e:
            wandb_logger.log({"csv_saved": False, "csv_path": csv_path, "error": str(e)})

        torch.cuda.empty_cache()
        gc.collect()
        return metrics


# ══════════════════════════════════════════════════════════════════════════════════
# main: 解析参数 → 构建模型 → 训练 → 测试
# ══════════════════════════════════════════════════════════════════════════════════

def main(args: ArgumentParser):
    """训练入口函数。"""

    # 设置随机种子 (保证可复现)
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    # bf16-mixed 精度下使用 medium 矩阵乘法精度 (更快)
    torch.set_float32_matmul_precision("medium")

    # 构建 run name 和 checkpoint 目录 (含 stage 标识)
    run_name = (f"{args.wandb_project}-{args.dataset_type}"
                f"-stage{args.stage}-{args.text_model_name.split('/')[-1]}")
    args.checkpoint_dir = (f"{args.checkpoint_dir}/{run_name}"
                           f"-{time.strftime('%Y%m%d-%H%M%S')}")

    # ── 构建 LightningModule ──
    model = DNALLMFineTuner(args)

    # ── Callbacks ──
    callbacks = [
        # 保存验证 loss 最低的 top-2 checkpoint
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-" + "{epoch:02d}-{val_loss_epoch:.4f}",
            save_top_k=2,
            monitor="val_loss_epoch",
            mode="min",
            save_last=True,  # 同时保存最后一个 epoch 的 checkpoint
        ),
        # 记录学习率变化
        LearningRateMonitor(logging_interval="step"),
    ]

    # ── Logger (wandb) ──
    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,
    )

    # ── Trainer ──
    # 关键参数:
    #   accumulate_grad_batches: 梯度累积, 等效增大 batch size (显存不够时的方案)
    #     effective_batch_size = batch_size × grad_accum × num_gpus
    #     如: 2 × 16 × 1 = 32
    #   precision="bf16-mixed": 混合精度训练 (forward/backward 用 bf16, 权重用 fp32)
    #   val_check_interval=1/3: 每个 epoch 做 3 次验证 (而不是每个 epoch 1 次)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,           # 最大训练 epoch 数
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=(
            "ddp"
            if args.strategy == "ddp"
            else DeepSpeedStrategy(           # DeepSpeed ZeRO Stage 2
                stage=2,                       # 分割 optimizer states + gradients
                offload_optimizer=False,       # optimizer 不 offload 到 CPU
                allgather_bucket_size=5e8,
                reduce_bucket_size=5e8)
        ),
        precision="bf16-mixed",               # bf16 混合精度
        callbacks=callbacks,
        logger=logger,
        deterministic=False,                  # 不强制确定性 (更快)
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,                  # 每 5 步记录一次日志
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,               # 梯度裁剪 (防止梯度爆炸)
        val_check_interval=1 / 3,             # 每个 epoch 验证 3 次
    )

    # ── 训练 ──
    # ckpt_path: 如果提供了 checkpoint, 从中恢复训练; 否则从头训练
    trainer.fit(model, ckpt_path=args.ckpt_path)

    # ── 测试 ──
    # 训练完成后在测试集上评估
    # ckpt_path="best" 表示用验证 loss 最低的 checkpoint
    trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")


if __name__ == "__main__":
    parser = ArgumentParser()

    # ── Model 参数 ──
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name", type=str,
                        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)

    # ── Training 参数 ──
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length_dna", type=int, default=1024)
    parser.add_argument("--max_length_text", type=int, default=1024)
    parser.add_argument("--truncate_dna_per_side", type=int, default=1024)
    parser.add_argument("--return_answer_in_batch", type=bool, default=False)

    # ── LoRA 参数 ──
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ── 路径参数 ──
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default="/model-weights")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")

    # ── Dataset 参数 ──
    parser.add_argument("--dataset_type", type=str,
                        choices=["kegg", "variant_effect_coding", "variant_effect_non_snv"],
                        default="kegg")
    parser.add_argument("--use_qwen_dna_collate_fn", type=bool, default=True)
    parser.add_argument("--kegg_data_dir_local", type=str, default="data/kegg")
    parser.add_argument("--kegg_data_dir_huggingface", type=str, default="wanglab/kegg")
    parser.add_argument("--variant_effect_coding_data_dir_huggingface", type=str,
                        default="wanglab/variant_effect_coding")
    parser.add_argument("--variant_effect_non_snv_data_dir_huggingface", type=str,
                        default="wanglab/variant_effect_non_snv")
    parser.add_argument("--merge_val_test_set", type=bool, default=False)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="1=阶段一(Pathogenic vs Benign), 2=阶段二(GOF vs LOF)")

    # ── Logging 参数 ──
    parser.add_argument("--wandb_project", type=str, default="nt-500m-qwen3-1.7b-finetune")
    parser.add_argument("--wandb_entity", type=str)

    args = parser.parse_args()
    main(args)
