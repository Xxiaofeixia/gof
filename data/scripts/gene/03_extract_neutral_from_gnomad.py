#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从本地 gnomAD exome 数据中提取中性候选变异。

本步骤采用偏保守的筛选策略：
  1. 只保留第 01 步目标基因范围内的变异。
  2. 使用服务器本地 gnomAD hg38 exome 频率数据。
  3. 保留相对常见的变异：AF >= 0.001。
  4. 排除已经出现在致病主表中的变异。
  5. 如果本地存在 ClinVar VCF，则排除 ClinVar 已记录的变异。
  6. 每个基因最多抽取 N 条，避免中性类别数量过大而主导训练。

输入：
  data/processed/01_target_genes.txt
  data/processed/01_pathogenic_variants_hg38.csv
  /gpfs/hpc/home/public/jclabadmin/Penetrance_annotation/Penetrance_annotation1/Coding_region/gencode.v47.annotation.gtf.gz
  /gpfs/hpc/home/public/jclabadmin/annotation/af/hg38_gnomad41_exome.txt.gz

输出：
  data/processed/03_neutral_variants_hg38.csv
  data/processed/03_neutral_to_annotate.vcf
  data/processed/03_neutral_audit_summary.csv
"""

from __future__ import annotations

import csv
import gzip
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE_DIR, "processed")

TARGET_GENES_IN = os.path.join(OUT_DIR, "01_target_genes.txt")
PATHOGENIC_IN = os.path.join(OUT_DIR, "01_pathogenic_variants_hg38.csv")
CLINVAR_VCF = os.path.join(BASE_DIR, "vep_results", "clinvar.vcf")

GTF_GZ = (
    "/gpfs/hpc/home/public/jclabadmin/Penetrance_annotation/"
    "Penetrance_annotation1/Coding_region/gencode.v47.annotation.gtf.gz"
)
GNOMAD_EXOME_GZ = "/gpfs/hpc/home/public/jclabadmin/annotation/af/hg38_gnomad41_exome.txt.gz"

NEUTRAL_OUT = os.path.join(OUT_DIR, "03_neutral_variants_hg38.csv")
VCF_OUT = os.path.join(OUT_DIR, "03_neutral_to_annotate.vcf")
SUMMARY_OUT = os.path.join(OUT_DIR, "03_neutral_audit_summary.csv")

MIN_AF = 0.001
MAX_PER_GENE = 20
RANDOM_SEED = 42
VALID_ALLELE_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def normalize_chrom(value: Any) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def normalize_allele(value: Any) -> str:
    return str(value).strip().upper()


def is_valid_vcf_allele(ref: str, alt: str) -> bool:
    return bool(VALID_ALLELE_RE.match(ref)) and bool(VALID_ALLELE_RE.match(alt))


def variant_key(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    return f"{normalize_chrom(chrom)}_{int(pos)}_{normalize_allele(ref)}_{normalize_allele(alt)}"


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text == ".":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_target_genes() -> set[str]:
    with open(TARGET_GENES_IN, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def load_pathogenic_keys() -> set[str]:
    df = pd.read_csv(PATHOGENIC_IN, dtype={"CHROM": "string", "POS": "Int64"}, low_memory=False)
    return {
        variant_key(row.CHROM, row.POS, row.REF, row.ALT)
        for row in df.itertuples(index=False)
    }


def load_clinvar_keys() -> set[str]:
    keys: set[str] = set()
    if not os.path.exists(CLINVAR_VCF):
        return keys
    with open(CLINVAR_VCF, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, ref = parts[0], parts[1], parts[3]
            for alt in parts[4].split(","):
                if is_valid_vcf_allele(ref, alt):
                    keys.add(variant_key(chrom, pos, ref, alt))
    return keys


def parse_gene_name(attributes: str) -> str | None:
    match = re.search(r'gene_name "([^"]+)"', attributes)
    if match:
        return match.group(1)
    match = re.search(r'gene_id "([^"]+)"', attributes)
    if match:
        return match.group(1).split(".")[0]
    return None


def load_gene_intervals(target_genes: set[str]) -> dict[str, list[tuple[int, int, str]]]:
    intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with gzip.open(GTF_GZ, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            gene_name = parse_gene_name(parts[8])
            if gene_name not in target_genes:
                continue
            chrom = normalize_chrom(parts[0])
            start = int(parts[3])
            end = int(parts[4])
            intervals[chrom].append((start, end, gene_name))

    for chrom in list(intervals):
        intervals[chrom].sort(key=lambda item: (item[0], item[1], item[2]))
    return intervals


def choose_af(row: dict[str, str]) -> float | None:
    for col in ("gnomad41_exome_AF", "gnomad41_exome_AF_raw", "gnomad41_exome_AF_grpmax"):
        af = parse_float(row.get(col))
        if af is not None:
            return af
    return None


def extract_candidates(
    intervals: dict[str, list[tuple[int, int, str]]],
    pathogenic_keys: set[str],
    clinvar_keys: set[str],
) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(RANDOM_SEED)
    by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter = Counter()
    interval_cursor: dict[str, int] = defaultdict(int)
    active_intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    def active_genes_for_variant(chrom: str, start: int, end: int) -> list[str]:
        chrom_intervals = intervals.get(chrom)
        if not chrom_intervals:
            return []

        cursor = interval_cursor[chrom]
        while cursor < len(chrom_intervals) and chrom_intervals[cursor][0] <= end:
            active_intervals[chrom].append(chrom_intervals[cursor])
            cursor += 1
        interval_cursor[chrom] = cursor

        active = [item for item in active_intervals[chrom] if item[1] >= start]
        active_intervals[chrom] = active
        return [gene for item_start, item_end, gene in active if item_start <= end and item_end >= start]

    with gzip.open(GNOMAD_EXOME_GZ, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and reader.fieldnames[0].startswith("#"):
            reader.fieldnames[0] = reader.fieldnames[0].lstrip("#")

        for row in reader:
            counters["gnomad_rows_seen"] += 1
            chrom = normalize_chrom(row["Chr"])
            start = int(row["Start"])
            end = int(row["End"])
            ref = normalize_allele(row["Ref"])
            alt = normalize_allele(row["Alt"])

            if not is_valid_vcf_allele(ref, alt):
                counters["skip_invalid_ref_alt_for_vcf"] += 1
                continue

            af = choose_af(row)
            if af is None or af < MIN_AF:
                counters["skip_low_or_missing_af"] += 1
                continue

            key = variant_key(chrom, start, ref, alt)
            if key in pathogenic_keys:
                counters["skip_pathogenic_overlap"] += 1
                continue
            if key in clinvar_keys:
                counters["skip_clinvar_recorded"] += 1
                continue

            genes = active_genes_for_variant(chrom, start, end)
            if not genes:
                counters["skip_outside_target_gene_interval"] += 1
                continue

            for gene in sorted(set(genes)):
                by_gene[gene].append({
                    "variant_key": key,
                    "CHROM": chrom,
                    "POS": start,
                    "END": end,
                    "REF": ref,
                    "ALT": alt,
                    "Gene": gene,
                    "LABEL": "Neutral",
                    "SOURCE": "gnomAD41_exome",
                    "AF": af,
                    "quality_status": "neutral_candidate_interval_pass",
                })
                counters["candidate_gene_assignments"] += 1

    sampled: list[dict[str, Any]] = []
    for gene, records in sorted(by_gene.items()):
        counters["genes_with_candidates_before_sampling"] += 1
        unique_records = {record["variant_key"]: record for record in records}
        records = list(unique_records.values())
        if len(records) > MAX_PER_GENE:
            records = rng.sample(records, MAX_PER_GENE)
            counters["genes_downsampled"] += 1
        sampled.extend(records)

    counters["neutral_records_after_sampling"] = len(sampled)
    counters["genes_with_neutral_after_sampling"] = len({record["Gene"] for record in sampled})
    return sampled, counters


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[int, str, int, str, str, str]:
        chrom = str(record["CHROM"])
        try:
            chrom_order = int(chrom)
        except ValueError:
            chrom_order = {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(chrom.upper(), 99)
        return chrom_order, chrom, int(record["POS"]), record["REF"], record["ALT"], record["Gene"]

    return sorted(records, key=key)


def deduplicate_records(records: list[dict[str, Any]], counters: Counter) -> list[dict[str, Any]]:
    """按变异主键去重，避免同一 gnomAD 变异因跨基因区间被重复用于训练。"""
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["variant_key"])
        if key not in unique:
            unique[key] = record

    counters["neutral_duplicate_variant_keys_removed"] = len(records) - len(unique)
    counters["neutral_records_after_variant_dedup"] = len(unique)
    return list(unique.values())


def write_vcf(records: list[dict[str, Any]]) -> None:
    with open(VCF_OUT, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##reference=GRCh38/hg38\n")
        handle.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Interval-assigned target gene\">\n")
        handle.write("##INFO=<ID=LABEL,Number=1,Type=String,Description=\"Neutral candidate label\">\n")
        handle.write("##INFO=<ID=AF,Number=1,Type=Float,Description=\"gnomAD v4.1 exome allele frequency\">\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for record in records:
            info = f"GENE={record['Gene']};LABEL=Neutral;AF={record['AF']:.8g}"
            handle.write(
                f"{record['CHROM']}\t{record['POS']}\t{record['variant_key']}\t"
                f"{record['REF']}\t{record['ALT']}\t.\tPASS\t{info}\n"
            )


def write_summary(counters: Counter, target_genes: set[str], intervals: dict[str, list[tuple[int, int, str]]]) -> None:
    genes_with_intervals = {gene for items in intervals.values() for _, _, gene in items}
    rows = [
        {"metric": "target_genes", "value": len(target_genes)},
        {"metric": "target_genes_with_gtf_interval", "value": len(genes_with_intervals)},
        {"metric": "min_af", "value": MIN_AF},
        {"metric": "max_per_gene", "value": MAX_PER_GENE},
        {"metric": "random_seed", "value": RANDOM_SEED},
    ]
    for key, value in counters.items():
        rows.append({"metric": key, "value": value})
    pd.DataFrame(rows).to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/6] 读取目标基因 ...", flush=True)
    target_genes = load_target_genes()
    print(f"target_genes={len(target_genes)}", flush=True)

    print("[2/6] 从致病主表和 ClinVar 读取需要排除的变异 ...", flush=True)
    pathogenic_keys = load_pathogenic_keys()
    clinvar_keys = load_clinvar_keys()
    print(f"pathogenic_keys={len(pathogenic_keys)} clinvar_keys={len(clinvar_keys)}", flush=True)

    print("[3/6] 从 GENCODE GTF 读取目标基因区间 ...", flush=True)
    intervals = load_gene_intervals(target_genes)
    genes_with_intervals = {gene for items in intervals.values() for _, _, gene in items}
    print(f"genes_with_gtf_interval={len(genes_with_intervals)}", flush=True)

    print("[4/6] 顺序扫描 gnomAD exome 并收集中性候选变异 ...", flush=True)
    records, counters = extract_candidates(intervals, pathogenic_keys, clinvar_keys)
    records = sort_records(records)
    records = deduplicate_records(records, counters)
    records = sort_records(records)

    print("[5/6] 写出 03 编号中性结果文件 ...", flush=True)
    pd.DataFrame(records).to_csv(NEUTRAL_OUT, index=False, encoding="utf-8-sig")
    write_vcf(records)
    write_summary(counters, target_genes, intervals)

    print("[6/6] 完成。", flush=True)
    print(f"NEUTRAL: {NEUTRAL_OUT}")
    print(f"VCF: {VCF_OUT}")
    print(f"SUMMARY: {SUMMARY_OUT}")
    print(pd.read_csv(SUMMARY_OUT).to_string(index=False))


if __name__ == "__main__":
    main()
