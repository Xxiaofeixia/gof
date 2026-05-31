"""
train_llm_only.py
纯LLM训练脚本（消融实验）。
基于 train_dna_qwen_vegg.py，去除所有DNA相关逻辑，不影响原始文件。
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

# ── 使用新建的纯LLM模型，不改动原始 dna_llm.py ──
from dna_llm_text_only import LLMOnlyModel, get_target_modules

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _make_collate_fn(processor, max_length_text, max_length_dna,
                     return_answer_in_batch, truncate_for_generation):
    return partial(
        qwen_dna_collate_fn,
        processor=processor,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
        return_answer_in_batch=return_answer_in_batch,
        truncate_for_generation=truncate_for_generation,
    )


def _make_processor(model):
    """纯LLM模式下 dna_tokenizer=None"""
    return DLProcessor(
        tokenizer=model.text_tokenizer,
        dna_tokenizer=None,
    )


class LLMOnlyFineTuner(pl.LightningModule):
    """纯LLM Fine-tuner，接口与 DNALLMFineTuner 一致，用于消融对比。"""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.text_model_name = self.hparams.text_model_name
        self.cache_dir = self.hparams.cache_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.text_model_finetune = self.hparams.text_model_finetune
        self.lora_rank = self.hparams.lora_rank
        self.lora_alpha = self.hparams.lora_alpha
        self.lora_dropout = self.hparams.lora_dropout
        self.max_length_dna = self.hparams.max_length_dna
        self.max_length_text = self.hparams.max_length_text
        self.return_answer_in_batch = self.hparams.return_answer_in_batch
        self.merge_val_test_set = self.hparams.merge_val_test_set
        self.dataset_type = self.hparams.dataset_type

        # 只加载LLM
        self.model = LLMOnlyModel(
            text_model_name=self.text_model_name,
            cache_dir=self.cache_dir,
            max_length_text=self.max_length_text,
        )
        self.text_model = self.model.text_model
        self.tokenizer = self.model.text_tokenizer

        self.lora_config = self._prep_for_training()

    def _prep_for_training(self) -> Optional[LoraConfig]:
      if self.text_model_finetune:
          target_modules = get_target_modules(self.model)
          lora_config = LoraConfig(
              r=self.lora_rank,
              lora_alpha=self.lora_alpha,
              lora_dropout=self.lora_dropout,
              target_modules=target_modules,
              init_lora_weights="gaussian",
              bias="none",
              task_type="CAUSAL_LM",
          )
          # ↓ 删掉 prepare_model_for_kbit_training 这行
          self.text_model = get_peft_model(self.text_model, lora_config)
          self.model.text_model = self.text_model
      else:
          for param in self.text_model.parameters():
              param.requires_grad = False
          lora_config = None
      return lora_config

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        # 🚀 FIX: 加个心跳打印，让你时刻知道模型没死！
        print(f"\n>>> [Heartbeat] Processing {prefix} split, Batch {batch_idx}...")
        
        if prefix == "test":
            return {"loss": torch.tensor(0.0, device=self.device)}

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None

        # dna_tokenized / batch_idx_map 传入但模型会忽略
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=None,
            batch_idx_map=None,
            labels=labels,
        )
        loss = outputs.loss

        # 偶尔打印生成样例
        if (prefix == "train" and self.global_step % 3000 == 0) or \
           (prefix == "val" and batch_idx % 300 == 0):
            try:
                assistant_start_marker = "<|im_start|>assistant\n"
                assistant_marker_tokens = self.tokenizer.encode(
                    assistant_start_marker, add_special_tokens=False)
                marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
                marker_len = len(assistant_marker_tokens)

                non_pad = (input_ids[0] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[0, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                if assistant_pos is not None:
                    gen_input_ids = input_ids[0:1, start_idx:assistant_pos + marker_len]
                    gen_attention_mask = attention_mask[0:1, start_idx:assistant_pos + marker_len]

                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            max_new_tokens=512,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                        )
                    user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                    generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()
                    print(f"\n=====[Sample {prefix} step={self.global_step}]=====")
                    print(f"Input:\n{user_input}")
                    print(f"Generation:\n{generation}")
                    del generated, gen_input_ids, gen_attention_mask
                    gc.collect()
            except Exception as e:
                print(f"Sample generation error: {e}")
                traceback.print_exc()

        current_lr = self.lr_schedulers().get_last_lr()[0] if prefix != "test" else 0
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(f"{prefix}_loss_epoch", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        if prefix != "test":
            self.log("lr", current_lr, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ── DataLoaders（与原脚本完全一致）────────────────────────────────────────

    def _load_dataset_splits(self, split: str):
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
            else:
                raw = (concatenate_datasets([dataset["test"], dataset["val"]])
                       if self.hparams.merge_val_test_set else dataset["test"])

        elif dtype == "variant_effect_coding":
            dataset = load_dataset(self.hparams.variant_effect_coding_data_dir_huggingface)
            cleaned = dataset.map(clean_variant_effect_example)

            def neutralize_prompt(example):
                example["question"] = "Based on the provided DNA sequences, please predict if this variant is Benign or Pathogenic."
                return example
            cleaned = cleaned.map(neutralize_prompt)

            dataset = cleaned.map(get_format_variant_effect_function(self.hparams.model_type))
            labels = []
            for sp, data in cleaned.items():
                labels.extend(data["answer"])
            train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            if split == "train":
                raw = train_val_split["train"]
            elif split == "val":
                raw = train_val_split["test"]
            else:
                raw = dataset["test"]

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
                raw = train_val_split["train"].map(get_format_variant_effect_function(self.hparams.model_type))
            elif split == "val":
                raw = train_val_split["test"].map(get_format_variant_effect_function(self.hparams.model_type))
            else:
                raw = dataset["test"].map(get_format_variant_effect_function(self.hparams.model_type))
        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        if self.hparams.truncate_dna_per_side:
            raw = raw.map(truncate_dna,
                          fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side})
        return raw, labels

    def _make_loader(self, split: str, shuffle: bool,
                     return_answer: bool, truncate_for_gen: bool) -> DataLoader:
        raw, labels = self._load_dataset_splits(split)
        self.labels = sorted(list(set(labels)))
        processor = _make_processor(self.model)
        collate_fn = _make_collate_fn(
            processor,
            max_length_text=self.max_length_text,
            max_length_dna=self.max_length_dna,
            return_answer_in_batch=return_answer,
            truncate_for_generation=truncate_for_gen,
        )
        return DataLoader(raw, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          collate_fn=collate_fn, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=False)

    def train_dataloader(self):
        return self._make_loader("train", shuffle=True,
                                 return_answer=self.return_answer_in_batch,
                                 truncate_for_gen=False)

    def val_dataloader(self):
        return self._make_loader("val", shuffle=False,
                                 return_answer=self.return_answer_in_batch,
                                 truncate_for_gen=False)

    def test_dataloader(self):
        return self._make_loader("test", shuffle=False,
                                 return_answer=True, truncate_for_gen=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run_name = (f"llm-only-{args.dataset_type}"
                f"-{args.text_model_name.split('/')[-1]}")
    args.checkpoint_dir = f"{args.checkpoint_dir}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    model = LLMOnlyFineTuner(args)

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

    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        
        # 🚀 FIX: 跳过导致卡死的 Sanity Check！
        num_sanity_val_steps=0, 
        
        strategy=(
            "ddp" if args.strategy == "ddp"
            else DeepSpeedStrategy(
                stage=2, 
                # 🚀 FIX: 开启 CPU 卸载，释放几个G的显存，硬扛OOM！
                offload_optimizer=True, 
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
    parser.add_argument("--model_type", type=str, default="llm")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--text_model_finetune", type=bool, default=True)

    # Training
    parser.add_argument("--seed", type=int, default=23)
    
    # 🚀 FIX: 极限显存压缩模式：batch size 1, 累积步数 16
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    
    parser.add_argument("--max_length_dna", type=int, default=512)
    parser.add_argument("--max_length_text", type=int, default=512)
    parser.add_argument("--truncate_dna_per_side", type=int, default=512)
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
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")

    # Dataset
    parser.add_argument("--dataset_type", type=str,
                        choices=["kegg", "variant_effect_coding", "variant_effect_non_snv"],
                        default="variant_effect_coding")
    parser.add_argument("--merge_val_test_set", type=bool, default=False)
    parser.add_argument("--kegg_data_dir_huggingface", type=str, default="wanglab/kegg")
    parser.add_argument("--variant_effect_coding_data_dir_huggingface", type=str,
                        default="wanglab/variant_effect_coding")
    parser.add_argument("--variant_effect_non_snv_data_dir_huggingface", type=str,
                        default="wanglab/variant_effect_non_snv")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="llm-only-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    main(args)