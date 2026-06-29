#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 04 步：用 ESM-2 masked language model 计算错义突变稳定性相关分数。

输入：
  data/processed/03_BIOREASON_with_RSA.csv
  data/processed/uniprot_mapping.csv
  data/processed/alphafold_pdbs/*.pdb

输出：
  data/processed/04_BIOREASON_with_ESM.csv

说明：
  ESM_DDG_Score = logit(mutant amino acid) - logit(wild-type amino acid)。
  仅对标准错义突变计算；其他类型返回 NaN，后续格式化阶段显示为 Unknown。
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter
from typing import Any

import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import warnings


warnings.filterwarnings("ignore")

BASE_DIR = "/gpfs/hpc/home/lijc/mapengtao/gof/data"
INPUT_CSV = os.path.join(BASE_DIR, "processed", "03_BIOREASON_with_RSA.csv")
MAPPING_CSV = os.path.join(BASE_DIR, "processed", "uniprot_mapping.csv")
PDB_DIR = os.path.join(BASE_DIR, "processed", "alphafold_pdbs")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed", "04_BIOREASON_with_ESM.csv")

MODEL_NAME = (
    "/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface/hub/"
    "models--facebook--esm2_t33_650M_UR50D/snapshots/"
    "08e4846e537177426273712802403f7ba8261b6c"
)
MAX_AA_WINDOW = 1022


def extract_pos(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def parse_aa(value: Any, index: int) -> str | None:
    text = str(value).strip()
    if pd.isna(value) or not text or text in {"-", "None", "nan"}:
        return None
    if "*" in text or "/" not in text:
        return None
    parts = text.split("/")
    if len(parts) < 2:
        return None
    aa = parts[index].strip()
    return aa[0].upper() if aa else None


def load_uniprot_mapping() -> dict[str, str]:
    mapping_df = pd.read_csv(MAPPING_CSV, low_memory=False)
    mapping: dict[str, str] = {}
    for row in mapping_df.itertuples(index=False):
        symbol = str(row.SYMBOL).strip()
        uid = str(row.UniProt_ID).strip()
        if symbol and uid and uid.lower() not in {"nan", "none", "-"}:
            mapping[symbol] = uid
    return mapping


def sequence_from_pdb(uid: str, counters: Counter) -> str | None:
    matched = glob.glob(os.path.join(PDB_DIR, f"AF-{uid}-F1-model_*.pdb"))
    if not matched:
        counters["missing_pdb"] += 1
        return None
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(uid, matched[0])
        builder = PPBuilder()
        sequence = "".join(str(peptide.get_sequence()) for peptide in builder.build_peptides(structure[0]))
        if not sequence:
            counters["empty_sequence"] += 1
            return None
        return sequence
    except Exception:
        counters["sequence_parse_failed"] += 1
        return None


def crop_sequence(sequence: str, pos_1based: int) -> tuple[str, int] | None:
    if pos_1based < 1 or pos_1based > len(sequence):
        return None
    if len(sequence) <= MAX_AA_WINDOW:
        return sequence, pos_1based

    half = MAX_AA_WINDOW // 2
    start_0based = max(0, pos_1based - 1 - half)
    end_0based = min(len(sequence), start_0based + MAX_AA_WINDOW)
    start_0based = max(0, end_0based - MAX_AA_WINDOW)
    local_pos_1based = pos_1based - start_0based
    return sequence[start_0based:end_0based], local_pos_1based


def calculate_esm_score(
    sequence: str | None,
    pos_1based: int | None,
    wt_aa: str | None,
    mut_aa: str | None,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    counters: Counter,
) -> float:
    if not sequence or pos_1based is None or wt_aa is None or mut_aa is None:
        counters["skip_invalid_input"] += 1
        return float("nan")

    cropped = crop_sequence(sequence, int(pos_1based))
    if cropped is None:
        counters["position_out_of_sequence"] += 1
        return float("nan")
    sequence_window, local_pos = cropped

    if sequence_window[local_pos - 1] != wt_aa:
        counters["wt_aa_mismatch"] += 1
        return float("nan")

    masked_sequence = sequence_window[: local_pos - 1] + tokenizer.mask_token + sequence_window[local_pos:]
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
    mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
    if wt_id == tokenizer.unk_token_id or mut_id == tokenizer.unk_token_id:
        counters["unknown_amino_acid_token"] += 1
        return float("nan")

    counters["assigned"] += 1
    return round(logits[0, mask_idx, mut_id].item() - logits[0, mask_idx, wt_id].item(), 4)


def main() -> None:
    print("启动第 04 步：ESM-2 稳定性分数计算。", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    model.eval()

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    uniprot_map = load_uniprot_mapping()
    pos_col = "Protein_position" if "Protein_position" in df.columns else "Amino_acids"
    df["mut_pos_tmp"] = df[pos_col].map(extract_pos)
    df["wt_aa_tmp"] = df["Amino_acids"].map(lambda value: parse_aa(value, 0))
    df["mut_aa_tmp"] = df["Amino_acids"].map(lambda value: parse_aa(value, -1))

    counters: Counter = Counter()
    sequence_cache: dict[str, str | None] = {}
    scores: list[float] = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        if idx % 500 == 0 or idx == len(df):
            print(f"ESM 行进度: {idx}/{len(df)}", flush=True)

        uid = uniprot_map.get(str(row.SYMBOL).strip())
        if not uid:
            counters["missing_uniprot"] += 1
            scores.append(float("nan"))
            continue
        if uid not in sequence_cache:
            sequence_cache[uid] = sequence_from_pdb(uid, counters)

        score = calculate_esm_score(
            sequence_cache[uid],
            row.mut_pos_tmp,
            row.wt_aa_tmp,
            row.mut_aa_tmp,
            tokenizer,
            model,
            device,
            counters,
        )
        scores.append(score)

    df["ESM_DDG_Score"] = scores
    df.drop(columns=["mut_pos_tmp", "wt_aa_tmp", "mut_aa_tmp"], inplace=True, errors="ignore")
    df.to_csv(OUTPUT_CSV, index=False)

    print("第 04 步完成。统计如下:", flush=True)
    for key, value in sorted(counters.items()):
        print(f"  {key}: {value}", flush=True)
    print(f"输出文件: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
