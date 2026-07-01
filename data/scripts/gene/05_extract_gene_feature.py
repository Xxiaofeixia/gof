#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析第 04 步 VEP 注释结果，合并 GOF/LOF/Neutral 标签，并提取 DNA 双序列。

输入：
  data/processed/04_pathogenic_vep_output.txt
  data/processed/04_neutral_vep_output.txt
  data/processed/01_pathogenic_variants_hg38.csv
  data/processed/03_neutral_variants_hg38.csv

输出：
  data/processed/BIOREASON_with_all_gene.csv
"""

from __future__ import annotations

import io
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import pysam


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

PATHOGENIC_VEP_TXT = os.path.join(PROCESSED_DIR, "04_pathogenic_vep_output.txt")
NEUTRAL_VEP_TXT = os.path.join(PROCESSED_DIR, "04_neutral_vep_output.txt")
PATHOGENIC_LABELS_CSV = os.path.join(PROCESSED_DIR, "01_pathogenic_variants_hg38.csv")
NEUTRAL_LABELS_CSV = os.path.join(PROCESSED_DIR, "03_neutral_variants_hg38.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "BIOREASON_with_all_gene.csv")

FASTA_PATH = "/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta"
FLANKING_BP = 2000

CONSEQUENCE_RANK = {
    "transcript_ablation": 1,
    "splice_acceptor_variant": 2,
    "splice_donor_variant": 3,
    "stop_gained": 4,
    "frameshift_variant": 5,
    "stop_lost": 6,
    "start_lost": 7,
    "inframe_insertion": 8,
    "inframe_deletion": 9,
    "missense_variant": 10,
    "protein_altering_variant": 11,
    "splice_region_variant": 12,
    "synonymous_variant": 13,
    "5_prime_UTR_variant": 14,
    "3_prime_UTR_variant": 15,
    "intron_variant": 16,
    "non_coding_transcript_exon_variant": 17,
    "upstream_gene_variant": 18,
    "downstream_gene_variant": 19,
    "intergenic_variant": 20,
}

TARGET_COLS = [
    "ID",
    "LABEL",
    "CHROM",
    "POS",
    "Location",
    "Allele",
    "Gene",
    "SYMBOL",
    "SOURCE",
    "Consequence",
    "DOMAINS",
    "Amino_acids",
    "Protein_position",
    "AF",
    "MAX_AF",
    "AlphaMissense_score",
    "CADD_phred",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "MutPred_score",
    "SpliceAI_pred_DS_AG",
    "SpliceAI_pred_DS_AL",
    "SpliceAI_pred_DS_DG",
    "SpliceAI_pred_DS_DL",
    "LoF",
    "LoF_filter",
    "LoF_flags",
    "LoF_info",
    "EXON",
    "REF",
    "ALT",
    "variant_key",
]


def read_vep_table(path: str, label_source: str) -> pd.DataFrame:
    """读取 VEP tab 输出，去掉 ## 元信息行，并统一 ID 列名。"""
    print(f"读取 VEP 注释结果: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [line for line in handle if not line.startswith("##")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    df.rename(columns={"#Uploaded_variation": "ID"}, inplace=True)
    df["vep_label_source"] = label_source
    return df


def consequence_rank(value: Any) -> int:
    consequences = str(value).split(",")
    ranks = [CONSEQUENCE_RANK.get(item.strip(), 99) for item in consequences if item.strip()]
    return min(ranks) if ranks else 99


def select_best_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """同一变异可能对应多个转录本，这里保留最适合下游建模的一行。"""
    work = df.copy()
    work["_consequence_rank"] = work["Consequence"].map(consequence_rank)
    work["_is_canonical"] = work.get("CANONICAL", "").astype(str).str.upper().eq("YES").astype(int)
    work["_has_amino_acids"] = ~work.get("Amino_acids", pd.Series(index=work.index, dtype=object)).isin(["-", "", "nan", "None"])
    work["_has_protein_pos"] = ~work.get("Protein_position", pd.Series(index=work.index, dtype=object)).isin(["-", "", "nan", "None"])
    work = work.sort_values(
        ["ID", "_consequence_rank", "_is_canonical", "_has_amino_acids", "_has_protein_pos"],
        ascending=[True, True, False, False, False],
    )
    work = work.drop_duplicates(subset=["ID"], keep="first")
    return work.drop(columns=["_consequence_rank", "_is_canonical", "_has_amino_acids", "_has_protein_pos"], errors="ignore")


def load_labels() -> pd.DataFrame:
    """合并致病标签表和中性标签表，统一使用 variant_key 作为 ID。"""
    pathogenic = pd.read_csv(PATHOGENIC_LABELS_CSV, low_memory=False)
    neutral = pd.read_csv(NEUTRAL_LABELS_CSV, low_memory=False)

    pathogenic = pathogenic.copy()
    neutral = neutral.copy()
    pathogenic["ID"] = pathogenic["variant_key"]
    neutral["ID"] = neutral["variant_key"]
    if "SOURCE_ID" not in neutral.columns:
        neutral["SOURCE_ID"] = ""
    if "quality_status" not in neutral.columns:
        neutral["quality_status"] = "neutral_candidate_interval_pass"

    keep_cols = ["ID", "variant_key", "LABEL", "CHROM", "POS", "REF", "ALT", "Gene", "SOURCE", "quality_status"]
    labels = pd.concat(
        [pathogenic.reindex(columns=keep_cols), neutral.reindex(columns=keep_cols)],
        ignore_index=True,
    )
    duplicate_count = int(labels["ID"].duplicated().sum())
    if duplicate_count:
        print(f"标签表发现重复变异 ID {duplicate_count} 条，按首次出现记录去重。")
        labels = labels.drop_duplicates(subset=["ID"], keep="first").copy()
    labels.rename(columns={"Gene": "label_gene"}, inplace=True)
    return labels


def parse_in_last_exon(value: Any) -> int | float:
    """从 EXON 字段解析是否位于最后一个外显子，无法解析时返回 NaN。"""
    try:
        current, total = str(value).strip().split("/")
        return 1 if int(current) == int(total) else 0
    except Exception:
        return np.nan


def numeric_or_nan(series: pd.Series) -> pd.Series:
    """解析 VEP/dbNSFP 数值列。

    dbNSFP 对同一变异/转录本常输出逗号分隔的多个值，例如 "0.887,0.887"。
    直接 pd.to_numeric 会把这种有效注释变成 NaN，因此这里逐项提取数值并取最大值。
    对 CADD、AlphaMissense、MutPred、SpliceAI、保守性和频率列，最大值代表最强证据。
    """
    def parse_one(value: Any) -> float:
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "-", ".", "None", "none", "nan", "NaN", "Unknown"}:
            return np.nan
        values: list[float] = []
        for part in re.split(r"[,;&|]", text):
            part = part.strip()
            if not part or part in {"-", "."}:
                continue
            try:
                values.append(float(part))
            except ValueError:
                continue
        return max(values) if values else np.nan

    return series.map(parse_one)


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """基于 hg38 参考基因组构建突变前/突变后 4kb DNA 序列。"""
    print(f"挂载参考基因组，提取上下游各 {FLANKING_BP}bp 的 DNA 双序列...")
    fasta = pysam.FastaFile(FASTA_PATH)
    ref_sequences: list[str] = []
    var_sequences: list[str] = []
    failed = 0

    for idx, row in df.iterrows():
        try:
            if "CHROM" in row and pd.notna(row["CHROM"]) and "POS" in row and pd.notna(row["POS"]):
                chrom_text = str(row["CHROM"]).strip()
                pos_1based = int(row["POS"])
            else:
                chrom_text, pos_text = str(row["Location"]).split(":")
                pos_1based = int(pos_text.split("-")[0])
            chrom = chrom_text if chrom_text.startswith("chr") else f"chr{chrom_text}"
            ref = str(row["REF"]).upper()
            alt = str(row["ALT"]).upper()
            ref_len = len(ref)

            start_0based = max(0, pos_1based - FLANKING_BP - 1)
            end_0based = pos_1based + FLANKING_BP + (ref_len - 1)
            ref_seq = fasta.fetch(reference=chrom, start=start_0based, end=end_0based).upper()

            center_ref = ref_seq[FLANKING_BP:FLANKING_BP + ref_len]
            if center_ref != ref:
                # 如果参考碱基与 FASTA 不一致，保留 ERROR，后续丢弃，避免构造错误 DNA 序列。
                raise ValueError(f"参考碱基不一致: fasta={center_ref}, table={ref}")

            var_seq = ref_seq[:FLANKING_BP] + alt + ref_seq[FLANKING_BP + ref_len:]
            ref_sequences.append(ref_seq[: FLANKING_BP * 2 + 1])
            var_sequences.append(var_seq[: FLANKING_BP * 2 + 1])
        except Exception:
            ref_sequences.append("ERROR")
            var_sequences.append("ERROR")
            failed += 1

        if idx > 0 and idx % 5000 == 0:
            print(f"  DNA 序列提取进度: {idx}/{len(df)}")

    df = df.copy()
    df["reference_sequence"] = ref_sequences
    df["variant_sequence"] = var_sequences
    df = df[(df["reference_sequence"] != "ERROR") & (df["variant_sequence"] != "ERROR")].copy()
    print(f"DNA 序列提取完成，失败并丢弃 {failed} 条，保留 {len(df)} 条。")
    return df


def main() -> None:
    print("启动第 05 步：解析 VEP 注释、合并标签、提取 DNA 双序列。")

    pathogenic_vep = read_vep_table(PATHOGENIC_VEP_TXT, "pathogenic")
    neutral_vep = read_vep_table(NEUTRAL_VEP_TXT, "neutral")
    vep = pd.concat([pathogenic_vep, neutral_vep], ignore_index=True)
    print(f"VEP 原始注释行数: {len(vep)}")

    vep = select_best_transcript(vep)
    print(f"按变异保留最佳转录本后: {len(vep)}")

    labels = load_labels()
    df = pd.merge(vep, labels, on="ID", how="inner", suffixes=("", "_label"))
    print(f"与 GOF/LOF/Neutral 标签表合并后: {len(df)}")

    df["variant_key"] = df["ID"]
    if "REF_label" in df.columns:
        df["REF"] = df["REF_label"]
    if "ALT_label" in df.columns:
        df["ALT"] = df["ALT_label"]
    if "SOURCE_label" in df.columns:
        df["SOURCE"] = df["SOURCE_label"]
    if "label_gene" in df.columns:
        missing_symbol = df["SYMBOL"].isna() | df["SYMBOL"].astype(str).isin(["-", "", "None", "nan"])
        df.loc[missing_symbol, "SYMBOL"] = df.loc[missing_symbol, "label_gene"]

    splice_cols = ["SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL"]
    for col in splice_cols:
        if col in df.columns:
            df[col] = numeric_or_nan(df[col])
    existing_splice = [col for col in splice_cols if col in df.columns]
    df["SpliceAI_DS_max"] = df[existing_splice].max(axis=1) if existing_splice else np.nan

    if "EXON" in df.columns:
        df["in_last_exon"] = df["EXON"].map(parse_in_last_exon)
    else:
        df["in_last_exon"] = np.nan

    score_cols = [
        "AlphaMissense_score",
        "CADD_phred",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "AF",
        "MAX_AF",
        "MutPred_score",
        "SpliceAI_DS_max",
    ]
    for col in score_cols:
        if col in df.columns:
            df[col] = numeric_or_nan(df[col])

    text_cols = ["Consequence", "DOMAINS", "Amino_acids", "SYMBOL", "Protein_position"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].replace("-", np.nan).fillna("None")

    df = build_sequences(df)

    output_cols = [col for col in TARGET_COLS if col in df.columns]
    output_cols += [col for col in ["SpliceAI_DS_max", "in_last_exon", "reference_sequence", "variant_sequence"] if col in df.columns]
    output_cols = list(dict.fromkeys(output_cols))
    df = df[output_cols].copy()
    duplicate_count = int(df["ID"].duplicated().sum())
    if duplicate_count:
        print(f"最终输出发现重复变异 ID {duplicate_count} 条，按首次出现记录去重。")
        df = df.drop_duplicates(subset=["ID"], keep="first").copy()

    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 70)
    print("第 05 步完成：基因级 VEP 特征和 DNA 双序列已生成。")
    print(f"输出文件: {OUTPUT_CSV}")
    print(f"最终有效记录数: {len(df)}")
    print(df["LABEL"].value_counts().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
