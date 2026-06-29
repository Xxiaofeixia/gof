#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据来源拆分正式实验数据集。

第 01 步生成的是完整整合审计表，包含：
  GOFCards + ClinVar_GOFLOF + HGMD2019_GOFLOF_HGMD2023_hg38

本步骤不重新做 liftover，也不重新合并原始文件，而是基于第 01 步的
variant-level 去重和冲突剔除结果，拆出更适合实验设计的数据集：

  1. 主实验致病集：HGMD-containing OR GOFCards-containing
     用于主训练，尽量保持文献/人工整理来源的一致性。

  2. ClinVar-only 外部验证致病集：
     只保留来源仅为 ClinVar，且不与 HGMD/GOFCards 重叠的变异。

  3. Mixed-full 致病集：
     保留第 01 步的全部混合致病集，用于补充实验。

  4. Shared-gene strict 数据集：
     只保留同时有 GOF 和 LOF 的基因，用于检查模型是否依赖基因名投机。

输入：
  data/processed/01_pathogenic_variants_hg38.csv
  data/processed/03_neutral_variants_hg38.csv

输出：
  data/processed/04_main_hgmd_gofcards_pathogenic.csv
  data/processed/04_main_hgmd_gofcards_dataset.csv
  data/processed/04_main_hgmd_gofcards_target_genes.txt
  data/processed/04_clinvar_unique_external_pathogenic.csv
  data/processed/04_mixed_full_pathogenic.csv
  data/processed/04_mixed_full_dataset.csv
  data/processed/04_shared_gene_strict_dataset.csv
  data/processed/04_source_aware_dataset_summary.csv
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE_DIR, "processed")

PATHOGENIC_IN = os.path.join(OUT_DIR, "01_pathogenic_variants_hg38.csv")
NEUTRAL_IN = os.path.join(OUT_DIR, "03_neutral_variants_hg38.csv")

MAIN_PATHOGENIC_OUT = os.path.join(OUT_DIR, "04_main_hgmd_gofcards_pathogenic.csv")
MAIN_DATASET_OUT = os.path.join(OUT_DIR, "04_main_hgmd_gofcards_dataset.csv")
MAIN_TARGET_GENES_OUT = os.path.join(OUT_DIR, "04_main_hgmd_gofcards_target_genes.txt")

CLINVAR_EXTERNAL_OUT = os.path.join(OUT_DIR, "04_clinvar_unique_external_pathogenic.csv")

MIXED_PATHOGENIC_OUT = os.path.join(OUT_DIR, "04_mixed_full_pathogenic.csv")
MIXED_DATASET_OUT = os.path.join(OUT_DIR, "04_mixed_full_dataset.csv")

SHARED_STRICT_OUT = os.path.join(OUT_DIR, "04_shared_gene_strict_dataset.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "04_source_aware_dataset_summary.csv")

HGMD_SOURCE = "HGMD2019_GOFLOF_HGMD2023_hg38"
GOFCARDS_SOURCE = "GOFCards"
CLINVAR_SOURCE = "ClinVar_GOFLOF"


def contains_source(series: pd.Series, source: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(source, regex=False)


def write_gene_list(genes: Iterable[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for gene in sorted({str(g).strip() for g in genes if str(g).strip()}):
            handle.write(f"{gene}\n")


def normalize_neutral_columns(neutral: pd.DataFrame) -> pd.DataFrame:
    neutral = neutral.copy()
    if "SOURCE_ID" not in neutral.columns:
        neutral["SOURCE_ID"] = ""
    if "original_build" not in neutral.columns:
        neutral["original_build"] = "hg38"
    if "original_chrom" not in neutral.columns:
        neutral["original_chrom"] = neutral["CHROM"]
    if "original_pos" not in neutral.columns:
        neutral["original_pos"] = neutral["POS"]
    if "note" not in neutral.columns:
        neutral["note"] = "gnomAD neutral candidate"
    return neutral


def combine_with_neutral(pathogenic: pd.DataFrame, neutral: pd.DataFrame) -> pd.DataFrame:
    genes = set(pathogenic["Gene"].dropna().astype(str))
    neutral_part = neutral[neutral["Gene"].astype(str).isin(genes)].copy()
    shared_columns = list(dict.fromkeys(list(pathogenic.columns) + list(neutral_part.columns)))
    return pd.concat(
        [pathogenic.reindex(columns=shared_columns), neutral_part.reindex(columns=shared_columns)],
        ignore_index=True,
    )


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    if len(df) == 0:
        return {}
    return {str(k): int(v) for k, v in df["LABEL"].value_counts().sort_index().items()}


def add_summary(rows: list[dict[str, object]], name: str, df: pd.DataFrame) -> None:
    rows.append({"dataset": name, "metric": "rows", "value": int(len(df))})
    rows.append({"dataset": name, "metric": "genes", "value": int(df["Gene"].nunique()) if "Gene" in df else 0})
    for label, count in label_counts(df).items():
        rows.append({"dataset": name, "metric": f"label:{label}", "value": count})


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/5] 读取第 01 步致病整合表和第 03 步中性候选表 ...")
    pathogenic = pd.read_csv(PATHOGENIC_IN, low_memory=False)
    neutral = normalize_neutral_columns(pd.read_csv(NEUTRAL_IN, low_memory=False))

    source = pathogenic["SOURCE"].fillna("").astype(str)
    has_hgmd = contains_source(source, HGMD_SOURCE)
    has_gofcards = contains_source(source, GOFCARDS_SOURCE)
    has_clinvar = contains_source(source, CLINVAR_SOURCE)

    print("[2/5] 拆分主实验致病集：HGMD-containing OR GOFCards-containing ...")
    main_pathogenic = pathogenic[has_hgmd | has_gofcards].copy()
    main_dataset = combine_with_neutral(main_pathogenic, neutral)
    write_gene_list(main_pathogenic["Gene"], MAIN_TARGET_GENES_OUT)

    print("[3/5] 拆分 ClinVar-only 外部验证致病集 ...")
    clinvar_external = pathogenic[has_clinvar & ~has_hgmd & ~has_gofcards].copy()

    print("[4/5] 生成 mixed-full 和 shared-gene strict 数据集 ...")
    mixed_pathogenic = pathogenic.copy()
    mixed_dataset = combine_with_neutral(mixed_pathogenic, neutral)

    gof_genes = set(pathogenic.loc[pathogenic["LABEL"].eq("GOF"), "Gene"].dropna().astype(str))
    lof_genes = set(pathogenic.loc[pathogenic["LABEL"].eq("LOF"), "Gene"].dropna().astype(str))
    shared_genes = gof_genes & lof_genes
    shared_pathogenic = pathogenic[pathogenic["Gene"].astype(str).isin(shared_genes)].copy()
    shared_neutral = neutral[neutral["Gene"].astype(str).isin(shared_genes)].copy()
    shared_columns = list(dict.fromkeys(list(shared_pathogenic.columns) + list(shared_neutral.columns)))
    shared_dataset = pd.concat(
        [shared_pathogenic.reindex(columns=shared_columns), shared_neutral.reindex(columns=shared_columns)],
        ignore_index=True,
    )

    print("[5/5] 写出 04 编号数据集和统计摘要 ...")
    main_pathogenic.to_csv(MAIN_PATHOGENIC_OUT, index=False, encoding="utf-8-sig")
    main_dataset.to_csv(MAIN_DATASET_OUT, index=False, encoding="utf-8-sig")
    clinvar_external.to_csv(CLINVAR_EXTERNAL_OUT, index=False, encoding="utf-8-sig")
    mixed_pathogenic.to_csv(MIXED_PATHOGENIC_OUT, index=False, encoding="utf-8-sig")
    mixed_dataset.to_csv(MIXED_DATASET_OUT, index=False, encoding="utf-8-sig")
    shared_dataset.to_csv(SHARED_STRICT_OUT, index=False, encoding="utf-8-sig")

    rows: list[dict[str, object]] = []
    for name, df in [
        ("main_hgmd_gofcards_pathogenic", main_pathogenic),
        ("main_hgmd_gofcards_dataset", main_dataset),
        ("clinvar_unique_external_pathogenic", clinvar_external),
        ("mixed_full_pathogenic", mixed_pathogenic),
        ("mixed_full_dataset", mixed_dataset),
        ("shared_gene_strict_dataset", shared_dataset),
    ]:
        add_summary(rows, name, df)
    rows.append({"dataset": "shared_gene_strict_dataset", "metric": "shared_gof_lof_genes", "value": len(shared_genes)})
    pd.DataFrame(rows).to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print("完成。")
    print(f"MAIN_PATHOGENIC: {MAIN_PATHOGENIC_OUT}")
    print(f"MAIN_DATASET: {MAIN_DATASET_OUT}")
    print(f"CLINVAR_EXTERNAL: {CLINVAR_EXTERNAL_OUT}")
    print(f"MIXED_DATASET: {MIXED_DATASET_OUT}")
    print(f"SHARED_STRICT: {SHARED_STRICT_OUT}")
    print(f"SUMMARY: {SUMMARY_OUT}")
    print(pd.read_csv(SUMMARY_OUT).to_string(index=False))


if __name__ == "__main__":
    main()
