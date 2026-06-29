#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将第 01 步生成的致病主表导出为 VEP 可读取的 VCF 文件。

输入：
  data/processed/01_pathogenic_variants_hg38.csv

输出：
  data/processed/02_pathogenic_to_annotate.vcf
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE_DIR, "processed")

MASTER_IN = os.path.join(OUT_DIR, "01_pathogenic_variants_hg38.csv")
VCF_OUT = os.path.join(OUT_DIR, "02_pathogenic_to_annotate.vcf")

VALID_INFO_RE = re.compile(r"[^A-Za-z0-9_.|:,+-]")


def normalize_chrom(value: Any) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def escape_info(value: Any) -> str:
    text = str(value).strip().replace(";", ",")
    return VALID_INFO_RE.sub("_", text)


def chrom_order(chrom: Any) -> tuple[int, str]:
    text = normalize_chrom(chrom)
    try:
        return int(text), text
    except ValueError:
        return {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(text.upper(), 99), text


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_chrom_order"] = work["CHROM"].map(lambda value: chrom_order(value)[0])
    work["_chrom_text"] = work["CHROM"].map(lambda value: chrom_order(value)[1])
    work = work.sort_values(["_chrom_order", "_chrom_text", "POS", "REF", "ALT"])
    return work.drop(columns=["_chrom_order", "_chrom_text"]).reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(MASTER_IN, dtype={"CHROM": "string", "POS": "Int64"}, low_memory=False)
    required = {"variant_key", "CHROM", "POS", "REF", "ALT", "Gene", "LABEL", "SOURCE", "quality_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {MASTER_IN}: {missing}")

    df = df[df["quality_status"].eq("pathogenic_strict_pass")].copy()
    df = sort_frame(df)

    with open(VCF_OUT, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##reference=GRCh38/hg38\n")
        handle.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"基因符号\">\n")
        handle.write("##INFO=<ID=LABEL,Number=1,Type=String,Description=\"致病效应标签：GOF 或 LOF\">\n")
        handle.write("##INFO=<ID=SOURCE,Number=.,Type=String,Description=\"合并后的数据来源\">\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for row in df.itertuples(index=False):
            chrom = normalize_chrom(row.CHROM)
            info = (
                f"GENE={escape_info(row.Gene)};"
                f"LABEL={escape_info(row.LABEL)};"
                f"SOURCE={escape_info(row.SOURCE)}"
            )
            handle.write(
                f"{chrom}\t{int(row.POS)}\t{row.variant_key}\t"
                f"{str(row.REF).upper()}\t{str(row.ALT).upper()}\t.\tPASS\t{info}\n"
            )

    print(f"完成。已写出 {len(df)} 条变异到 {VCF_OUT}")


if __name__ == "__main__":
    main()
