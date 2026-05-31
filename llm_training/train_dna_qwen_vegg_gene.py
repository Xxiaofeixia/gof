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

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _make_collate_fn(processor, max_length_text, max_length_dna,
                     return_answer_in_batch, truncate_for_generation):
    """Helper: build a collate_fn with the given settings."""
    return partial(
        qwen_dna_collate_fn,
        processor=processor,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
        return_answer_in_batch=return_answer_in_batch,
        truncate_for_generation=truncate_for_generation,
    )


def _make_processor(model):
    return DLProcessor(
        tokenizer=model.text_tokenizer,
        dna_tokenizer=model.dna_tokenizer,
    )


class DNALLMFineTuner(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning DNA-LLM models."""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

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
        self.lora_config = self._prep_for_training()

    def _prep_for_training(self) -> LoraConfig:
        if self.dna_model_finetune:
            pass
        else:
            if self.dna_is_evo2:
                for param in self.dna_model.model.parameters():
                    param.requires_grad = False
            else:
                for param in self.dna_model.parameters():
                    param.requires_grad = False

        if self.text_model_finetune:
            target_modules = get_target_modules(self)
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                init_lora_weights=True,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_config)
        else:
            for param in self.text_model.parameters():
                param.requires_grad = False

        for param in self.dna_projection.parameters():
            param.requires_grad = True

        return lora_config

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        if prefix == "test":
            return {"loss": torch.tensor(0.0, device=self.device)}

        input_ids = batch["input_ids"].to(self.device).long()
        attention_mask = batch["attention_mask"].to(self.device).long()
        labels = batch["labels"].to(self.device).long() if "labels" in batch else None
        
        dna_tokenized = batch.get("dna_tokenized")
        if dna_tokenized is not None:
            dna_tokenized = dna_tokenized.to(self.device)
            # 专门针对 DNA 模型的输入强制转整型，彻底击碎 FloatTensor 报错！
            dna_tokenized["input_ids"] = dna_tokenized["input_ids"].long()
            if "attention_mask" in dna_tokenized:
                dna_tokenized["attention_mask"] = dna_tokenized["attention_mask"].long()
                
        batch_idx_map = batch.get("batch_idx_map")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
            labels=labels,
        )

        loss = outputs.loss

        if (prefix == "train" and (self.global_step % 3000 == 0)) or \
           (prefix == "val" and (batch_idx % 300 == 0)):
            try:
                example_idx = 0
                print(f"\n=== Sample Generation (step {self.global_step} / "
                      f"{self.trainer.estimated_stepping_batches}) ===")

                assistant_start_marker = "<|im_start|>assistant\n"
                assistant_marker_tokens = self.tokenizer.encode(
                    assistant_start_marker, add_special_tokens=False)
                marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
                marker_len = len(assistant_marker_tokens)

                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                matches = []
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        matches.append(pos)
                        break
                assistant_pos = matches[0] if matches else None

                if assistant_pos is not None:
                    gen_input_ids = input_ids[
                        example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                    gen_attention_mask = attention_mask[
                        example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]

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

                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            dna_tokenized=example_dna_data,
                            batch_idx_map=example_batch_map,
                            max_new_tokens=2000,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                        )

                    user_input = self.tokenizer.decode(
                        gen_input_ids[0], skip_special_tokens=False).strip()
                    generation = self.tokenizer.decode(
                        generated[0], skip_special_tokens=False).strip()

                    del generated, gen_input_ids, gen_attention_mask, example_dna_data, example_batch_map
                    gc.collect()

                    print(f"=====[Sample {prefix} {batch_idx}]=====")
                    print(f"=====[User input]=====\n{user_input}")
                    print(f"=====[Complete generation]=====\n{generation}")

                    ground_truth = ""
                    if labels is not None:
                        valid_label_pos = (labels[example_idx] != -100).nonzero(as_tuple=True)[0]
                        if len(valid_label_pos) > 0:
                            if valid_label_pos[0] >= assistant_pos + marker_len:
                                ground_truth = self.tokenizer.decode(
                                    input_ids[example_idx, valid_label_pos],
                                    skip_special_tokens=False).strip()
                                print(f"=====[Ground truth]=====\n{ground_truth}")

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

        if prefix != "test":
            current_lr = self.lr_schedulers().get_last_lr()[0]
        else:
            current_lr = 0

        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        self.log(f"{prefix}_loss_epoch", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        if prefix != "test":
            self.log("lr", current_lr, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, prefix="val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ──────────────────────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────────────────────

    def _load_dataset_splits(self, split: str):
        """
        Load and format the dataset for the requested split.

        split: "train" | "val" | "test"

        Returns (raw_dataset, labels_list)
        """
        dtype = self.hparams.dataset_type

        if dtype == "kegg":
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
            dataset = load_dataset("csv", data_files="/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_gene_Dataset_LLM_Ready.csv")
            
            # 2. 执行安全清洗与 DNA-LLM 格式化 (极其关键！)
            
            cleaned = dataset.map(clean_variant_effect_example, load_from_cache_file=False)
            formatted = cleaned.map(get_format_variant_effect_function(self.hparams.model_type), load_from_cache_file=False)
            
            # 3. 获取格式化后的数据
            raw_data = formatted["train"]
            labels = [example["answer"] for example in raw_data]
            
            # 4. 严谨的科学划分：80% 训练, 10% 验证, 10% 测试
            split_1 = raw_data.train_test_split(test_size=0.2, seed=42)
            split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
            
            if split == "train":
                raw = split_1["train"]
            elif split == "val":
                raw = split_2["train"]
            else:  # test
                raw = split_2["test"]

        elif dtype == "variant_effect_non_snv":
            dataset = load_dataset(self.hparams.variant_effect_non_snv_data_dir_huggingface)
            dataset = dataset.map(clean_variant_effect_non_snv_example)
            cleaned = dataset.map(clean_variant_effect_example)
            labels = []
            for sp, data in cleaned.items():
                labels.extend(data["answer"])
            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
            # 同样从 train 中切出 10% 作为 val
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

        if self.hparams.truncate_dna_per_side:
            raw = raw.map(truncate_dna,
                          fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side})

        return raw, labels

    def train_dataloader(self) -> DataLoader:
        raw, labels = self._load_dataset_splits("train")
        self.labels = sorted(list(set(labels)))

        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=self.return_answer_in_batch,
            # train: keep full sequence (with assistant answer) so labels are valid
            truncate_for_generation=False,
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        raw, labels = self._load_dataset_splits("val")
        self.labels = sorted(list(set(labels)))

        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=self.return_answer_in_batch,
            # val: keep full sequence so loss is valid
            truncate_for_generation=False,
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        raw, labels = self._load_dataset_splits("test")
        self.labels = sorted(list(set(labels)))

        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            # test: must return answer for evaluation, and truncate to prompt-only
            return_answer_in_batch=True,
            truncate_for_generation=True,
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Test evaluation  (multi-class accuracy)
    # ──────────────────────────────────────────────────────────────────────────

    def on_test_epoch_end(self):
        """
        Generate text for every test example and compute accuracy.

        Works for any number of classes (kegg has 37, VEP has 2).
        Accuracy = fraction of examples where ground-truth label appears
        in the generated response (case-insensitive substring match).

        For binary datasets (variant_effect_*) also reports precision /
        recall / F1 using labels[1] as the positive class.
        """
        wandb_logger = self.logger.experiment
        wandb_logger.log({"test_progress": 0.0, "status": "starting test generation"})

        self.model.eval()

        test_dataloader = self.test_dataloader()
        total_batches = len(test_dataloader)
        is_binary = len(self.labels) == 2

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
        processed_batches = 0
        generations = []

        for batch_idx, batch in enumerate(test_dataloader):
            wandb_logger.log({
                "test_progress": batch_idx / total_batches,
                "status": f"processing batch {batch_idx}/{total_batches}",
            })

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            answer = batch["answer"]
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
            for example_idx in range(input_ids.size(0)):

                if total_examples % 10 == 0:
                    current_acc = correct_count / max(1, total_examples)
                    wandb_logger.log({"examples_processed": total_examples,
                                      "current_accuracy": current_acc})

                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id
                           ).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                if assistant_pos is None:
                    continue

                gen_input_ids = input_ids[
                    example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                gen_attention_mask = attention_mask[
                    example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]

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

                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        dna_tokenized=example_dna_data,
                        batch_idx_map=example_batch_map,
                        max_new_tokens=2000,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                    )

                user_input = self.tokenizer.decode(
                    gen_input_ids[0], skip_special_tokens=False).strip()
                generation = self.tokenizer.decode(
                    generated[0], skip_special_tokens=False).strip()

                ground_truth = answer[example_idx].strip()

                # ── accuracy (works for any number of classes) ──────────────
                # 只检查 </think> 之后的答案部分，避免推理过程中偶然命中
                if "</think>" in generation:
                    answer_part = generation.split("</think>")[-1]
                else:
                    answer_part = generation
                hit = ground_truth.lower() in answer_part.lower()
                total_examples += 1
                examples_in_batch += 1
                if hit:
                    correct_count += 1

                # ── binary metrics ───────────────────────────────────────────
                prediction_category = "correct" if hit else "wrong"
                if is_binary:
                    is_pos = ground_truth.lower() == pos_label.lower()
                    is_neg = ground_truth.lower() == neg_label.lower()
                    if is_pos and hit:
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

        # ── final metrics ────────────────────────────────────────────────────
        accuracy = correct_count / max(total_examples, 1)
        metrics = {
            "test_accuracy": accuracy,
            "total_examples_processed": total_examples,
            "correct_count": correct_count,
            "test_status": "completed",
        }

        if is_binary:
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
            })
            summary = (
                f"Test Results:\n"
                f"Total: {total_examples}  Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}\n"
                f"TP={true_positives} FP={false_positives} "
                f"TN={true_negatives} FN={false_negatives}"
            )
        else:
            from sklearn.metrics import classification_report, accuracy_score
            
            y_true_3class = []
            y_pred_3class = []
            y_true_binary = []
            y_pred_binary = []
            
            # 1. 智能提取模型意图，并同时生成 3分类 和 2分类 的标签
            for g in generations:
                ans = g["generation"].lower()
                truth = g["ground_truth"]
                
                # 记录真实标签 (3分类 & 2分类)
                y_true_3class.append(truth)
                y_true_binary.append("Pathogenic" if "Pathogenic" in truth else "Benign")
                
                # 解析预测标签 (3分类)
                if "gof" in ans or "gain-of-function" in ans:
                    pred_3 = self.labels[0] if "gof" in self.labels[0].lower() else "Pathogenic; Gain-of-Function (GOF)"
                    pred_2 = "Pathogenic"
                elif "lof" in ans or "loss-of-function" in ans:
                    pred_3 = self.labels[1] if "lof" in self.labels[1].lower() else "Pathogenic; Loss-of-Function (LOF)"
                    pred_2 = "Pathogenic"
                elif "neutral" in ans or "benign" in ans:
                    pred_3 = self.labels[2] if "neutral" in self.labels[2].lower() else "Benign; Neutral"
                    pred_2 = "Benign"
                else:
                    pred_3 = "Unknown/Invalid"
                    pred_2 = "Unknown/Invalid"
                
                y_pred_3class.append(pred_3)
                y_pred_binary.append(pred_2)
            
            # 2. 计算第一阶段：致病性预测表现 (VPatho 风格)
            bin_acc = accuracy_score(y_true_binary, y_pred_binary)
            bin_report = classification_report(y_true_binary, y_pred_binary, zero_division=0)
            
            # 3. 计算第二阶段：效应细分表现 (GOF / LOF / Neutral)
            multi_acc = accuracy_score(y_true_3class, y_pred_3class)
            multi_report = classification_report(y_true_3class, y_pred_3class, zero_division=0)
            
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
                f"=======================================================\n"
            )

        print(summary)
        wandb_logger.log({**metrics, "test_summary": summary})

        # ── save generations table ───────────────────────────────────────────
        if generations:
            columns = list(generations[0].keys())
            wandb_logger.log({
                f"test_generations_{time.strftime('%Y%m%d-%H%M%S')}": wandb.Table(
                    columns=columns,
                    data=[[g.get(c, "") for c in columns] for g in generations],
                )
            })

        # ── save CSV ─────────────────────────────────────────────────────────
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


def main(args: ArgumentParser):
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run_name = (f"{args.wandb_project}-{args.dataset_type}"
                f"-{args.text_model_name.split('/')[-1]}")
    args.checkpoint_dir = (f"{args.checkpoint_dir}/{run_name}"
                           f"-{time.strftime('%Y%m%d-%H%M%S')}")

    model = DNALLMFineTuner(args)

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

    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=(
            "ddp"
            if args.strategy == "ddp"
            else DeepSpeedStrategy(
                stage=2, offload_optimizer=False,
                allgather_bucket_size=5e8, reduce_bucket_size=5e8)
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
        val_check_interval=1 / 3,
    )

    trainer.fit(model, ckpt_path=args.ckpt_path)
    trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name", type=str,
                        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)

    # Training
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

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default="/model-weights")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")

    # Dataset
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

    # Logging
    parser.add_argument("--wandb_project", type=str, default="nt-500m-qwen3-1.7b-finetune")
    parser.add_argument("--wandb_entity", type=str)

    args = parser.parse_args()
    main(args)