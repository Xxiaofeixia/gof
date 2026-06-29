#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 03 步：用 AlphaFold PDB 和 DSSP 计算二级结构与相对溶剂可及性。

输入：
  data/processed/01_BIOREASON_with_DBs.csv
  data/processed/uniprot_mapping.csv
  data/processed/alphafold_pdbs/*.pdb

输出：
  data/processed/03_BIOREASON_with_RSA.csv

实现要点：
  1. 只对标准错义突变尝试提取蛋白结构特征。
  2. 同一个 UniProt 只运行一次 DSSP，并缓存该蛋白所有残基的 SS/RSA。
  3. 找不到结构、位置越界或 DSSP 失败时记为 Unknown，不中断整体流程。
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter
from typing import Any

import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import warnings


warnings.simplefilter("ignore", BiopythonWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "processed", "01_BIOREASON_with_DBs.csv")
MAPPING_CSV = os.path.join(BASE_DIR, "processed", "uniprot_mapping.csv")
PDB_DIR = os.path.join(BASE_DIR, "processed", "alphafold_pdbs")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed", "03_BIOREASON_with_RSA.csv")

DSSP_BIN = "/gpfs/hpc/home/lijc/mapengtao/miniconda3/envs/dssp_env/bin/mkdssp"


def extract_pos(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def is_valid_missense(value: Any) -> bool:
    text = str(value).strip()
    if pd.isna(value) or not text or text in {"-", "None", "nan"}:
        return False
    if "*" in text or "/" not in text:
        return False
    return True


def load_uniprot_mapping() -> dict[str, str]:
    mapping_df = pd.read_csv(MAPPING_CSV, low_memory=False)
    mapping: dict[str, str] = {}
    for row in mapping_df.itertuples(index=False):
        symbol = str(row.SYMBOL).strip()
        uid = str(row.UniProt_ID).strip()
        if symbol and uid and uid.lower() not in {"nan", "none", "-"}:
            mapping[symbol] = uid
    return mapping


def find_pdb(uid: str) -> str | None:
    matched = glob.glob(os.path.join(PDB_DIR, f"AF-{uid}-F1-model_*.pdb"))
    return matched[0] if matched else None


def build_dssp_cache(uid: str, counters: Counter) -> dict[int, tuple[str, float]] | None:
    pdb_file = find_pdb(uid)
    if not pdb_file:
        counters["missing_pdb"] += 1
        return None
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(uid, pdb_file)
        dssp = DSSP(structure[0], pdb_file, dssp=DSSP_BIN)
        residue_features: dict[int, tuple[str, float]] = {}
        for key in dssp.keys():
            residue_id = key[1]
            residue_number = residue_id[1] if isinstance(residue_id, tuple) else residue_id
            residue_features[int(residue_number)] = (dssp[key][2], dssp[key][3])
        counters["dssp_success"] += 1
        return residue_features
    except Exception:
        counters["dssp_failed"] += 1
        return None


def main() -> None:
    print("启动第 03 步：DSSP 二级结构和 RSA 计算。", flush=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    uniprot_map = load_uniprot_mapping()

    pos_col = "Protein_position" if "Protein_position" in df.columns else "Amino_acids"
    df["mut_pos_tmp"] = df[pos_col].map(extract_pos)
    df["is_missense_tmp"] = df["Amino_acids"].map(is_valid_missense)

    needed_symbols = sorted({
        str(row.SYMBOL).strip()
        for row in df[df["is_missense_tmp"]].itertuples(index=False)
        if str(row.SYMBOL).strip()
    })
    needed_uids = sorted({
        uniprot_map[symbol]
        for symbol in needed_symbols
        if symbol in uniprot_map
    })
    print(f"需要 DSSP 的 UniProt 数: {len(needed_uids)}", flush=True)

    counters: Counter = Counter()
    dssp_cache: dict[str, dict[int, tuple[str, float]] | None] = {}
    for idx, uid in enumerate(needed_uids, start=1):
        dssp_cache[uid] = build_dssp_cache(uid, counters)
        if idx % 50 == 0 or idx == len(needed_uids):
            print(f"DSSP 蛋白进度: {idx}/{len(needed_uids)}", flush=True)

    secondary_structure: list[str] = []
    rsa_values: list[str | float] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        if idx % 5000 == 0 or idx == len(df):
            print(f"写回行进度: {idx}/{len(df)}", flush=True)

        if not row.is_missense_tmp or pd.isna(row.mut_pos_tmp):
            secondary_structure.append("Unknown")
            rsa_values.append("Unknown")
            counters["non_missense_or_no_position"] += 1
            continue

        uid = uniprot_map.get(str(row.SYMBOL).strip())
        if not uid:
            secondary_structure.append("Unknown")
            rsa_values.append("Unknown")
            counters["missing_uniprot"] += 1
            continue

        residue_features = dssp_cache.get(uid)
        if not residue_features:
            secondary_structure.append("Unknown")
            rsa_values.append("Unknown")
            continue

        feature = residue_features.get(int(row.mut_pos_tmp))
        if feature is None:
            secondary_structure.append("Unknown")
            rsa_values.append("Unknown")
            counters["position_not_in_structure"] += 1
            continue

        ss, rsa = feature
        secondary_structure.append(ss)
        rsa_values.append(rsa)
        counters["assigned"] += 1

    df["Secondary_Structure"] = secondary_structure
    df["AlphaFold_RSA"] = rsa_values
    df.drop(columns=["mut_pos_tmp", "is_missense_tmp"], inplace=True, errors="ignore")
    df.to_csv(OUTPUT_CSV, index=False)

    print("第 03 步完成。统计如下:", flush=True)
    for key, value in sorted(counters.items()):
        print(f"  {key}: {value}", flush=True)
    print(f"输出文件: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
