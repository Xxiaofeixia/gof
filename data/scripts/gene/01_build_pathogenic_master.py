#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建正式主实验致病 GOF/LOF 主表和目标基因列表。

本版第 01 步只合并两个偏文献/人工整理来源：
  1. GOFCards：人工整理的 GOF 变异
  2. HGMD-based GOF/LOF：已经 liftover 到 hg38 并与 HGMD2023 hg38 坐标严格匹配的变异

注意：
  ClinVar-based GOF/LOF 不再混入主训练集。
  ClinVar 后续应作为外部验证集单独处理，以降低 source bias。

输入：
  data/raw/gofcards_data_download.xlsx
  data/processed/goflof_HGMD2019_liftover_hg38_strict_pass.csv
  data/reference/hg19ToHg38.over.chain.gz

输出：
  data/processed/01_pathogenic_variants_hg38.csv
  data/processed/01_target_genes.txt
  data/processed/01_pathogenic_removed_or_review.csv
  data/processed/01_pathogenic_audit_summary.csv
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
from pyliftover import LiftOver


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUT_DIR = os.path.join(BASE_DIR, "processed")
REF_DIR = os.path.join(BASE_DIR, "reference")

GOFCARDS_XLSX = os.path.join(RAW_DIR, "gofcards_data_download.xlsx")
HGMD_STRICT_CSV = os.path.join(OUT_DIR, "goflof_HGMD2019_liftover_hg38_strict_pass.csv")
CHAIN_FILE = os.path.join(REF_DIR, "hg19ToHg38.over.chain.gz")

MASTER_OUT = os.path.join(OUT_DIR, "01_pathogenic_variants_hg38.csv")
TARGET_GENES_OUT = os.path.join(OUT_DIR, "01_target_genes.txt")
REVIEW_OUT = os.path.join(OUT_DIR, "01_pathogenic_removed_or_review.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "01_pathogenic_audit_summary.csv")

VALID_ALLELE_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def normalize_chrom(value: Any) -> str:
    """统一染色体写法：去掉 chr 前缀。"""
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def normalize_allele(value: Any) -> str:
    """统一 REF/ALT 写法：转成大写，缺失值转为空字符串。"""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def is_valid_vcf_allele(ref: str, alt: str) -> bool:
    """只保留 VCF 标准碱基写法，过滤 '-' 等非标准等位基因。"""
    return bool(VALID_ALLELE_RE.match(ref)) and bool(VALID_ALLELE_RE.match(alt))


def variant_key(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    """构建变异唯一键：CHROM_POS_REF_ALT。"""
    return f"{normalize_chrom(chrom)}_{int(pos)}_{normalize_allele(ref)}_{normalize_allele(alt)}"


def liftover_1based(lo: LiftOver, chrom: Any, pos_1based: Any) -> tuple[str | None, int | None, str]:
    """pyliftover 内部使用 0-based 坐标；本函数输入和输出都保持 1-based 坐标。"""
    try:
        pos_int = int(pos_1based)
    except (TypeError, ValueError):
        return None, None, "invalid_pos"

    hits = lo.convert_coordinate(f"chr{normalize_chrom(chrom)}", pos_int - 1)
    if not hits:
        return None, None, "liftover_unmapped"

    target_chrom, target_pos_0based = hits[0][0], hits[0][1]
    status = "liftover_ok" if len(hits) == 1 else "liftover_multi_first_used"
    return normalize_chrom(target_chrom), int(target_pos_0based) + 1, status


def make_record(
    *,
    source: str,
    source_id: str,
    label: str,
    gene: str,
    chrom: Any,
    pos: Any,
    ref: Any,
    alt: Any,
    original_build: str,
    original_chrom: Any,
    original_pos: Any,
    status: str = "ok",
    note: str = "",
) -> dict[str, Any]:
    """把不同来源的变异整理成统一列结构。"""
    ref_norm = normalize_allele(ref)
    alt_norm = normalize_allele(alt)
    chrom_norm = normalize_chrom(chrom)
    pos_int = int(pos)
    return {
        "variant_key": variant_key(chrom_norm, pos_int, ref_norm, alt_norm),
        "CHROM": chrom_norm,
        "POS": pos_int,
        "REF": ref_norm,
        "ALT": alt_norm,
        "Gene": str(gene).strip(),
        "LABEL": str(label).strip().upper(),
        "SOURCE": source,
        "SOURCE_ID": str(source_id),
        "original_build": original_build,
        "original_chrom": normalize_chrom(original_chrom),
        "original_pos": str(original_pos),
        "quality_status": status,
        "note": note,
    }


def append_or_review(records: list[dict[str, Any]], review: list[dict[str, Any]], record: dict[str, Any]) -> None:
    """通过基础质量检查的记录进入主表，否则进入 review 文件。"""
    if record["LABEL"] not in {"GOF", "LOF"}:
        record["quality_status"] = "review_invalid_label"
        review.append(record)
        return
    if not record["Gene"] or record["Gene"].lower() in {"nan", "none", "-"}:
        record["quality_status"] = "review_missing_gene"
        review.append(record)
        return
    if not is_valid_vcf_allele(record["REF"], record["ALT"]):
        record["quality_status"] = "review_invalid_ref_alt_for_vcf"
        review.append(record)
        return
    records.append(record)


def load_gofcards(lo: LiftOver, review: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """读取 GOFCards，并将 hg19 坐标 liftover 到 hg38。"""
    df = pd.read_excel(GOFCARDS_XLSX)
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        source_id = str(row.get("Order numbe", idx + 2))
        chrom38, pos38, lift_status = liftover_1based(lo, row.get("chr"), row.get("hg19start"))
        if chrom38 is None or pos38 is None:
            review.append({
                "variant_key": "",
                "CHROM": "",
                "POS": "",
                "REF": normalize_allele(row.get("ref")),
                "ALT": normalize_allele(row.get("alt")),
                "Gene": str(row.get("genesymbol", "")).strip(),
                "LABEL": "GOF",
                "SOURCE": "GOFCards",
                "SOURCE_ID": source_id,
                "original_build": "hg19",
                "original_chrom": normalize_chrom(row.get("chr")),
                "original_pos": str(row.get("hg19start")),
                "quality_status": lift_status,
                "note": "GOFCards hg19start liftover failed",
            })
            continue

        record = make_record(
            source="GOFCards",
            source_id=source_id,
            label="GOF",
            gene=row.get("genesymbol", ""),
            chrom=chrom38,
            pos=pos38,
            ref=row.get("ref"),
            alt=row.get("alt"),
            original_build="hg19",
            original_chrom=row.get("chr"),
            original_pos=row.get("hg19start"),
            status=lift_status,
        )
        append_or_review(records, review, record)
    return records


def load_hgmd_strict(review: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """读取已经通过 hg19->hg38 liftover 和 HGMD2023 hg38 严格校验的 HGMD 结果。"""
    df = pd.read_csv(
        HGMD_STRICT_CSV,
        dtype={
            "ID": "string",
            "LABEL": "string",
            "GENE": "string",
            "CHROM_hg38_HGMD2023": "string",
            "POS_hg38_HGMD2023": "Int64",
            "REF": "string",
            "ALT": "string",
        },
        low_memory=False,
    )
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        record = make_record(
            source="HGMD2019_GOFLOF_HGMD2023_hg38",
            source_id=row.get("ID", idx),
            label=row.get("LABEL"),
            gene=row.get("GENE", ""),
            chrom=row.get("CHROM_hg38_HGMD2023"),
            pos=row.get("POS_hg38_HGMD2023"),
            ref=row.get("REF"),
            alt=row.get("ALT"),
            original_build="hg38",
            original_chrom=row.get("CHROM_hg38_HGMD2023"),
            original_pos=row.get("POS_hg38_HGMD2023"),
            status="hgmd_strict_pass",
        )
        append_or_review(records, review, record)
    return records


def collapse_same_label(records: list[dict[str, Any]], review: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 variant_key 合并同标签重复记录；同一变异 GOF/LOF 冲突则剔除。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["variant_key"]].append(record)

    collapsed: list[dict[str, Any]] = []
    for key, items in grouped.items():
        labels = sorted({item["LABEL"] for item in items})
        if len(labels) > 1:
            for item in items:
                bad = item.copy()
                bad["quality_status"] = "removed_gof_lof_conflict"
                bad["note"] = f"variant_key has conflicting labels: {','.join(labels)}"
                review.append(bad)
            continue

        first = items[0].copy()
        first["SOURCE"] = ";".join(sorted({item["SOURCE"] for item in items}))
        first["SOURCE_ID"] = ";".join(sorted({item["SOURCE_ID"] for item in items if item["SOURCE_ID"]}))
        first["quality_status"] = "pathogenic_strict_pass"
        first["note"] = f"merged_same_label_records={len(items)}"
        collapsed.append(first)
    return collapsed


def sort_key(row: pd.Series) -> tuple[int, str, int, str, str]:
    """按染色体和坐标稳定排序。"""
    chrom = str(row["CHROM"])
    try:
        chrom_order = int(chrom)
    except ValueError:
        chrom_order = {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(chrom.upper(), 99)
    return chrom_order, chrom, int(row["POS"]), str(row["REF"]), str(row["ALT"])


def write_summary(raw_records: list[dict[str, Any]], master: pd.DataFrame, review: pd.DataFrame) -> None:
    """写出第 01 步审计统计，方便检查每个来源和标签的数量。"""
    rows: list[dict[str, Any]] = [
        {"metric": "source_policy", "value": "HGMD + GOFCards only; ClinVar held out"},
        {"metric": "raw_records_before_dedup_and_conflict", "value": len(raw_records)},
        {"metric": "master_records_after_dedup_and_conflict", "value": len(master)},
        {"metric": "removed_or_review_records", "value": len(review)},
        {"metric": "target_genes", "value": master["Gene"].nunique() if len(master) else 0},
    ]

    for label, count in Counter(record["LABEL"] for record in raw_records).items():
        rows.append({"metric": f"raw_label:{label}", "value": count})
    for source, count in Counter(record["SOURCE"] for record in raw_records).items():
        rows.append({"metric": f"raw_source:{source}", "value": count})
    if len(master):
        for label, count in master["LABEL"].value_counts().sort_index().items():
            rows.append({"metric": f"master_label:{label}", "value": int(count)})
        exploded_sources = master["SOURCE"].str.split(";").explode()
        for source, count in exploded_sources.value_counts().sort_index().items():
            rows.append({"metric": f"master_source_contains:{source}", "value": int(count)})
    if len(review):
        for status, count in review["quality_status"].value_counts().sort_index().items():
            rows.append({"metric": f"review_status:{status}", "value": int(count)})

    pd.DataFrame(rows).to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lo = LiftOver(CHAIN_FILE)

    review_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []

    print("[1/4] 读取并标准化 GOFCards GOF ...")
    raw_records.extend(load_gofcards(lo, review_records))
    print("[2/4] 读取 HGMD strict-pass GOF/LOF ...")
    raw_records.extend(load_hgmd_strict(review_records))

    print("[3/4] 按 variant_key 去重，并删除 GOF/LOF 冲突变异 ...")
    collapsed = collapse_same_label(raw_records, review_records)
    master = pd.DataFrame(collapsed)
    if len(master):
        master["_sort_tuple"] = master.apply(sort_key, axis=1)
        master = master.sort_values("_sort_tuple").drop(columns=["_sort_tuple"]).reset_index(drop=True)

    review = pd.DataFrame(review_records)

    print("[4/4] 写出 01 编号结果文件 ...")
    master.to_csv(MASTER_OUT, index=False, encoding="utf-8-sig")
    if len(master):
        genes = master["Gene"].dropna().astype(str).str.strip()
        target_genes = sorted(genes.loc[genes.ne("")].unique())
    else:
        target_genes = []
    with open(TARGET_GENES_OUT, "w", encoding="utf-8") as handle:
        for gene in target_genes:
            handle.write(f"{gene}\n")

    review.to_csv(REVIEW_OUT, index=False, encoding="utf-8-sig")
    write_summary(raw_records, master, review)

    print("完成。")
    print(f"MASTER: {MASTER_OUT}")
    print(f"TARGET_GENES: {TARGET_GENES_OUT}")
    print(f"REVIEW: {REVIEW_OUT}")
    print(f"SUMMARY: {SUMMARY_OUT}")
    print(pd.read_csv(SUMMARY_OUT).to_string(index=False))


if __name__ == "__main__":
    main()
