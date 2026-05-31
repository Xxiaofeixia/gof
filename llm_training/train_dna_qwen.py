"""
SFT 监督微调训练脚本 —— 本项目最主要、最完整的训练入口。
=====================================================================================
基于 PyTorch Lightning 框架，使用 LoRA 高效微调 DNA-LLM 多模态模型。

训练流程概览:
  1. 加载数据集 (KEGG / Variant Effect Coding / Variant Effect Non-SNV)
  2. 构建 DNALLMModel (DNA encoder + Projection + Text LLM)
  3. 设置 LoRA（只对文本 LLM 的线性层做低秩适配）
  4. 使用 SFT 方式训练：模型的 loss = 交叉熵（只在 assistant 回复位置计算）
  5. 通过 W&B 记录训练指标和生成样本

三种数据集:
  - kegg:                     KEGG 生物通路推理（配对DNA序列 + 问题 + 答案 + 推理过程）
  - variant_effect_coding:    编码区变异效应预测（致病性 Benign/Pathogenic）
  - variant_effect_non_snv:   非SNV变异效应预测（插入缺失等）

两种模型模式:
  - model_type="dna-llm":  DNA序列作为 embedding 输入（多模态）
  - model_type="llm":      DNA序列作为文本拼接到问题中（纯文本对比实验）

关键设计:
  - LoRA:   只微调文本 LLM + dna_projection，DNA encoder 冻结
  - Labels: 只有 assistant 的回复参与 loss 计算（user 输入被忽略）
  - 定期生成样本并记录到 W&B 观察模型学习进展
=====================================================================================
"""

import csv
import gc
import os
import time
import traceback
from argparse import ArgumentParser
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

# CUDA 多进程兼容性设置
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DNALLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning 模块 —— DNA-LLM 的微调封装。

    继承 pl.LightningModule，自动处理:
    - 训练/验证/测试的 step 逻辑
    - optimizer 和 scheduler 的配置
    - DataLoader 的创建
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # ── 保存超参数 ──
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

        # ── Step 1: 构建 DNALLMModel ──
        # 这个模型包含: DNA encoder + Projection + Text LLM
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

        self.text_model = self.model.text_model
        self.dna_model = self.model.dna_model
        self.dna_projection = self.model.dna_projection
        self.tokenizer = self.model.text_tokenizer

        # ── Step 2: 配置 LoRA ──
        self.lora_config = self._prep_for_training()

    def _prep_for_training(self) -> LoraConfig:
        """
        准备训练：冻结/解冻各个组件，设置 LoRA。

        训练策略:
        - DNA encoder:      冻结（不训练）
        - Projection layer: 始终可训练（bridge layer）
        - Text LLM:         用 LoRA 训练（只训练低秩适配器）
        """
        # ── 冻结 DNA encoder ──
        if self.dna_model_finetune:
            pass  # 不冻结，全部可训练
        else:
            # 默认情况：DNA encoder 参数全部冻结
            if self.dna_is_evo2:
                for param in self.dna_model.model.parameters():
                    param.requires_grad = False
            else:
                for param in self.dna_model.parameters():
                    param.requires_grad = False

        # ── 文本 LLM 用 LoRA ──
        if self.text_model_finetune:
            # 获取所有需要应用 LoRA 的线性层名称
            target_modules = get_target_modules(self)

            lora_config = LoraConfig(
                r=self.lora_rank,            # LoRA 秩（低秩矩阵的维度）
                lora_alpha=self.lora_alpha,   # LoRA 缩放因子
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )

            # 准备文本模型进行 k-bit 训练 + 添加 LoRA 适配器
            self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_config)
        else:
            # 不训练文本模型：全部冻结
            for param in self.text_model.parameters():
                param.requires_grad = False

        # ── Projection 层始终可训练 ──
        # 这是 DNA 和文本之间的桥梁，必须训练
        for param in self.dna_projection.parameters():
            param.requires_grad = True

        return lora_config

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        通用训练/验证/测试步骤。

        流程:
        1. 从 batch 中取出 input_ids, attention_mask, labels, dna_tokenized
        2. 送入模型前向传播
        3. 计算 loss（交叉熵，只在 labels!=-100 的位置计算）
        4. 定期生成样本（训练每3000步，验证每300步）
        5. 记录指标到 W&B
        """
        # 测试模式直接返回0（测试在 on_test_epoch_end 自己处理）
        if prefix == "test":
            return {"loss": torch.tensor(0.0, device=self.device)}

        # 将 batch 数据移到 GPU
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        dna_tokenized = batch.get("dna_tokenized")
        if dna_tokenized is not None:
            dna_tokenized = dna_tokenized.to(self.device)
        batch_idx_map = batch.get("batch_idx_map")

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
            labels=labels,
        )

        loss = outputs.loss

        # ── 定期生成样本观察模型进展 ──
        if (prefix == "train" and (self.global_step % 3000 == 0)) or (prefix == "val" and (batch_idx % 300 == 0)):
            try:
                example_idx = 0
                print(
                    f"\n=== Sample Generation (step {self.global_step} / {self.trainer.estimated_stepping_batches}) ==="
                )

                # 找到 assistant 标记位置
                assistant_start_marker = "<|im_start|>assistant\n"
                assistant_marker_tokens = self.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
                marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
                marker_len = len(assistant_marker_tokens)

                # 找到第一个非 padding token
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                # 查找 assistant 标记
                matches = []
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos : pos + marker_len] == marker_tensor):
                        matches.append(pos)
                        break

                assistant_pos = matches[0] if matches else None

                if assistant_pos is not None:
                    # 截取到 assistant 标记位置（只给模型 user 输入，让其自己生成回复）
                    gen_input_ids = input_ids[
                        example_idx : example_idx + 1, start_idx : assistant_pos + marker_len
                    ]
                    gen_attention_mask = attention_mask[
                        example_idx : example_idx + 1, start_idx : assistant_pos + marker_len
                    ]

                    # 为这个样本准备好对应的 DNA 数据
                    example_dna_data = None
                    example_batch_map = None

                    if dna_tokenized is not None and batch_idx_map is not None:
                        example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]

                        if len(example_indices) > 0:
                            example_dna_data = BatchEncoding(
                                {
                                    "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                                    "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                                }
                            )
                            example_batch_map = [0] * len(example_indices)

                    # 生成文本
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
                        )

                    # 解码并输出
                    user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                    generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                    del generated, gen_input_ids, gen_attention_mask, example_dna_data, example_batch_map
                    gc.collect()

                    print(f"=====[Sample {prefix} {batch_idx}]=====")
                    print(f"=====[User input]=====\n{user_input}")
                    print(f"=====[Complete generation]=====\n{generation}")

                    # 如果有 ground truth，也打印
                    ground_truth = ""
                    if labels is not None:
                        valid_label_pos = (labels[example_idx] != -100).nonzero(as_tuple=True)[0]

                        if len(valid_label_pos) > 0:
                            if valid_label_pos[0] >= assistant_pos + marker_len:
                                ground_truth = self.tokenizer.decode(
                                    input_ids[example_idx, valid_label_pos], skip_special_tokens=False
                                ).strip()
                                print(f"=====[Ground truth]=====\n{ground_truth}")

                    # 记录到 W&B
                    timestamp = time.time()
                    step_id = f"gen_{self.global_step}-{timestamp}"
                    wandb_logger = self.logger.experiment
                    wandb_logger.log(
                        {
                            step_id: wandb.Table(
                                columns=["timestamp", "prefix", "batch_idx", "user_input", "generation", "ground_truth"],
                                data=[[timestamp, prefix, batch_idx, user_input, generation, ground_truth]],
                            )
                        }
                    )

                    del user_input, generation, ground_truth
                    torch.cuda.empty_cache()
                    gc.collect()

                else:
                    print("No assistant marker found in the input sequence")

            except Exception as e:
                print(f"Error during sample generation: {str(e)}")
                traceback.print_exc()

        # ── 获取当前学习率 ──
        if prefix != "test":
            current_lr = self.lr_schedulers().get_last_lr()[0]
        else:
            current_lr = 0

        # ── 记录指标到 W&B ──
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(f"{prefix}_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if prefix != "test":
            self.log("lr", current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, prefix="val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。

        - 优化器: AdamW
        - 调度器: Cosine with warmup（前10%步数线性warmup，然后cosine衰减）
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)  # 前10%步数做warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        """创建训练 DataLoader"""
        # 根据 dataset_type 加载不同的数据集
        if self.hparams.dataset_type == "kegg":
            dataset = load_dataset(self.hparams.kegg_data_dir_huggingface)
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

        elif self.hparams.dataset_type == "variant_effect_coding":
            dataset = load_dataset(self.hparams.variant_effect_coding_data_dir_huggingface)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

        elif self.hparams.dataset_type == "variant_effect_non_snv":
            dataset = load_dataset(self.hparams.variant_effect_non_snv_data_dir_huggingface)
            dataset = dataset.map(clean_variant_effect_non_snv_example)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            train_dataset = train_dataset.map(get_format_variant_effect_function(self.hparams.model_type))

        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")

        # 构建 processor 和 collate function
        processor = DLProcessor(
            tokenizer=self.model.text_tokenizer,
            dna_tokenizer=self.model.dna_tokenizer,
        )

        collate_fn = partial(
            qwen_dna_collate_fn,
            processor=processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=self.return_answer_in_batch,
            truncate_for_generation=False,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        """创建验证 DataLoader —— 逻辑与 train_dataloader 类似，但用验证集"""
        # ... (与 train_dataloader 类似，但 data split 不同)
        # 此处省略详细注释以节省篇幅，结构与 train_dataloader 完全一致
        if self.hparams.dataset_type == "kegg":
            dataset = load_dataset(self.hparams.kegg_data_dir_huggingface)
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))

            if self.hparams.merge_val_test_set:
                val_dataset = concatenate_datasets([dataset['test'], dataset['val']])
            else:
                val_dataset = dataset["val"]

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

        elif self.hparams.dataset_type == "variant_effect_coding":
            dataset = load_dataset(self.hparams.variant_effect_coding_data_dir_huggingface)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            val_dataset = dataset["test"]

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

        elif self.hparams.dataset_type == "variant_effect_non_snv":
            dataset = load_dataset(self.hparams.variant_effect_non_snv_data_dir_huggingface)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(clean_variant_effect_non_snv_example)

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
            val_dataset = dataset["test"]

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            val_dataset = val_dataset.map(get_format_variant_effect_function(self.hparams.model_type))

        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")

        processor = DLProcessor(
            tokenizer=self.model.text_tokenizer,
            dna_tokenizer=self.model.dna_tokenizer,
        )

        collate_fn = partial(
            qwen_dna_collate_fn,
            processor=processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=self.return_answer_in_batch,
            truncate_for_generation=False,
        )

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        """测试 DataLoader —— 复用 val_dataloader"""
        return self.val_dataloader()

    def on_test_epoch_end(self):
        """
        测试阶段结束时的回调 —— 对每个测试样本生成文本并计算准确率。

        计算二分类指标:
        - Accuracy, Precision, Recall, F1
        - TP, FP, TN, FN

        结果保存到 CSV 文件并记录到 W&B。
        """
        wandb_logger = self.logger.experiment
        wandb_logger.log({"test_progress": 0.0, "status": "starting test generation"})

        self.model.eval()

        test_dataloader = self.test_dataloader()
        total_batches = len(test_dataloader)

        # 确定正负标签
        neg_label = self.labels[0]
        pos_label = self.labels[1]

        wandb_logger.log({
            "positive_label": pos_label,
            "negative_label": neg_label
        })
        print(f"Using labels - Positive: '{pos_label}', Negative: '{neg_label}'")

        # 初始化计数器
        total_examples = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        processed_batches = 0
        generations = []

        for batch_idx, batch in enumerate(test_dataloader):
            wandb_logger.log({
                "test_progress": batch_idx / total_batches,
                "status": f"processing batch {batch_idx}/{total_batches}"
            })

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            answer = batch["answer"]
            dna_tokenized = batch.get("dna_tokenized")
            if dna_tokenized is not None:
                dna_tokenized = dna_tokenized.to(self.device)
            batch_idx_map = batch.get("batch_idx_map")

            assistant_start_marker = "<|im_start|>assistant\n"
            assistant_marker_tokens = self.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
            marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
            marker_len = len(assistant_marker_tokens)

            for example_idx in range(input_ids.size(0)):
                if total_examples % 10 == 0:
                    current_accuracy = (true_positives + true_negatives) / max(1, total_examples)
                    wandb_logger.log({
                        "examples_processed": total_examples,
                        "current_accuracy": current_accuracy
                    })

                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                if assistant_pos is not None:
                    gen_input_ids = input_ids[example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                    gen_attention_mask = attention_mask[example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]

                    example_dna_data = None
                    example_batch_map = None

                    if dna_tokenized is not None and batch_idx_map is not None:
                        example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]

                        if example_indices:
                            example_dna_data = BatchEncoding({
                                "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                                "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                            })
                            example_batch_map = [0] * len(example_indices)

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
                        )

                    user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                    generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                    # 获取 ground truth
                    ground_truth = answer[example_idx]
                    if ";" in ground_truth:
                        ground_truth = ground_truth.split(";")[0]

                    # 判断预测是否正确
                    is_positive_example = ground_truth.lower() == pos_label.lower()
                    is_negative_example = ground_truth.lower() == neg_label.lower()
                    generation_contains_ground_truth = ground_truth.lower() in generation.lower()

                    total_examples += 1

                    # 更新混淆矩阵
                    if is_positive_example and generation_contains_ground_truth:
                        true_positives += 1
                    elif is_positive_example and not generation_contains_ground_truth:
                        false_negatives += 1
                    elif is_negative_example and generation_contains_ground_truth:
                        true_negatives += 1
                    elif is_negative_example and not generation_contains_ground_truth:
                        false_positives += 1

                    prediction_category = (
                        "TP" if is_positive_example and generation_contains_ground_truth else
                        "FN" if is_positive_example and not generation_contains_ground_truth else
                        "TN" if is_negative_example and generation_contains_ground_truth else
                        "FP"
                    )

                    generations.append({
                        "batch_idx": batch_idx,
                        "example_idx": example_idx,
                        "user_input": user_input,
                        "generation": generation,
                        "ground_truth": ground_truth,
                        "contains_ground_truth": generation_contains_ground_truth,
                        "is_positive_example": is_positive_example,
                        "prediction_category": prediction_category
                    })

                    torch.cuda.empty_cache()
                    gc.collect()

        # 计算最终指标
        accuracy = (true_positives + true_negatives) / max(total_examples, 1)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # 记录最终结果
        wandb_logger.log({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "total_examples_processed": total_examples,
            "test_status": "completed"
        })

        # 保存 CSV
        model_name = self.hparams.text_model_name.split('/')[-1]
        if self.hparams.ckpt_path:
            csv_path = os.path.join(self.hparams.ckpt_path, f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv")
        else:
            csv_path = os.path.join(self.hparams.checkpoint_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv")

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if generations:
                    writer = csv.DictWriter(f, fieldnames=generations[0].keys())
                    writer.writeheader()
                    for g in generations:
                        writer.writerow(g)
                    wandb_logger.log({"csv_saved": True, "csv_path": csv_path})
        except Exception as e:
            wandb_logger.log({"csv_saved": False, "csv_path": csv_path, "error": str(e)})

        summary = (
            f"Test Results Summary:\n"
            f"Total examples: {total_examples}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}"
        )
        print(summary)
        wandb_logger.log({"test_summary": summary})

        torch.cuda.empty_cache()
        gc.collect()

        return {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }


def main(args: ArgumentParser):
    """
    主函数 —— 设置训练环境并启动训练。

    训练策略:
    - 支持 DDP（多GPU数据并行）或 DeepSpeed Stage 2
    - 使用 bf16 混合精度
    - Checkpoint 保存最佳2个模型（基于验证 loss）
    - 通过 W&B 记录所有指标
    """
    # 随机种子
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # 运行名称
    run_name = f"{args.wandb_project}-{args.dataset_type}-{args.text_model_name.split('/')[-1]}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    # 构建模型
    model = DNALLMFineTuner(args)

    # 回调：Checkpoint 保存 + 学习率监控
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-" + "{epoch:02d}-{val_loss_epoch:.4f}",
            save_top_k=2,
            monitor="val_loss_epoch",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # W&B Logger
    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,
    )

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=(
            "ddp"
            if args.strategy == "ddp"
            else DeepSpeedStrategy(stage=2, offload_optimizer=False, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
        ),
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        val_check_interval=1 / 3,  # 每个epoch做3次验证
    )

    # 开始训练
    trainer.fit(model, ckpt_path=args.ckpt_path)
    trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")


if __name__ == "__main__":
    parser = ArgumentParser()

    # ── 模型配置 ──
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name", type=str, default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)

    # ── 训练参数 ──
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

    # ── 路径配置 ──
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default="/model-weights")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")

    # ── 数据集配置 ──
    parser.add_argument("--dataset_type", type=str, choices=["kegg", "variant_effect_coding", "variant_effect_non_snv"], default="kegg")
    parser.add_argument("--use_qwen_dna_collate_fn", type=bool, default=True)
    parser.add_argument("--kegg_data_dir_huggingface", type=str, default="wanglab/kegg")
    parser.add_argument("--variant_effect_coding_data_dir_huggingface", type=str, default="wanglab/variant_effect_coding")
    parser.add_argument("--variant_effect_non_snv_data_dir_huggingface", type=str, default="wanglab/variant_effect_non_snv")
    parser.add_argument("--merge_val_test_set", type=bool, default=False)

    # ── 日志配置 ──
    parser.add_argument("--wandb_project", type=str, default="nt-500m-qwen3-1.7b-finetune")
    parser.add_argument("--wandb_entity", type=str)

    args = parser.parse_args()

    main(args)
