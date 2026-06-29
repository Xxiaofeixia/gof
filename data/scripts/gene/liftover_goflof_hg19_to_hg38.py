#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GOF/LOF HGMD2019 数据中的 hg19 坐标 liftover 到 hg38，
并与 HGMDpro202301 hg38 文件按 ID + CHROM + POS 做严格匹配。

输入：
  data/raw/goflof_HGMD2019_v032021.csv
  data/raw/HGMDpro202301hg38PMID.xlsx
  data/reference/hg19ToHg38.over.chain.gz

输出：
  data/processed/goflof_HGMD2019_liftover_hg38_match_HGMD2023.csv
  data/processed/goflof_HGMD2019_liftover_hg38_strict_pass.csv
  data/processed/goflof_HGMD2019_liftover_hg38_needs_review.csv
  data/processed/goflof_HGMD2019_liftover_summary.csv
"""

from __future__ import annotations

import os
from collections import Counter

import pandas as pd
from pyliftover import LiftOver


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "raw")
REF_DIR = os.path.join(BASE_DIR, "reference")
OUT_DIR = os.path.join(BASE_DIR, "processed")

GOFLOF_CSV = os.path.join(RAW_DIR, "goflof_HGMD2019_v032021.csv")
HGMD_XLSX = os.path.join(RAW_DIR, "HGMDpro202301hg38PMID.xlsx")
CHAIN_FILE = os.path.join(REF_DIR, "hg19ToHg38.over.chain.gz")

FULL_OUT = os.path.join(OUT_DIR, "goflof_HGMD2019_liftover_hg38_match_HGMD2023.csv")
STRICT_OUT = os.path.join(OUT_DIR, "goflof_HGMD2019_liftover_hg38_strict_pass.csv")
REVIEW_OUT = os.path.join(OUT_DIR, "goflof_HGMD2019_liftover_hg38_needs_review.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "goflof_HGMD2019_liftover_summary.csv")


def normalize_chrom(value: object) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def chrom_for_liftover(value: object) -> str:
    chrom = normalize_chrom(value)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def first_unique_liftover(lo: LiftOver, chrom: object, pos_1based: object) -> dict[str, object]:
    """将 1-based hg19 单点坐标转换为 1-based hg38 坐标。"""
    try:
        pos_int = int(pos_1based)
    except (TypeError, ValueError):
        return {
            "CHROM_hg38_liftover": pd.NA,
            "POS_hg38_liftover": pd.NA,
            "liftover_status": "invalid_pos",
            "liftover_hits": 0,
        }

    # pyliftover 使用 0-based 坐标。输入 POS-1，输出再 +1 还原为 1-based。
    hits = lo.convert_coordinate(chrom_for_liftover(chrom), pos_int - 1)
    if not hits:
        return {
            "CHROM_hg38_liftover": pd.NA,
            "POS_hg38_liftover": pd.NA,
            "liftover_status": "unmapped",
            "liftover_hits": 0,
        }

    # 对单点变异，理论上应唯一映射；若多重映射，先取第一个并标记复核。
    target_chrom, target_pos_0based = hits[0][0], hits[0][1]
    status = "mapped_unique" if len(hits) == 1 else "mapped_multi_first_used"
    return {
        "CHROM_hg38_liftover": normalize_chrom(target_chrom),
        "POS_hg38_liftover": int(target_pos_0based) + 1,
        "liftover_status": status,
        "liftover_hits": len(hits),
    }


def make_match_status(row: pd.Series) -> str:
    if not bool(row["ID_found_in_HGMD2023"]):
        return "ID_not_found_in_HGMD2023"
    if row["liftover_status"] in {"invalid_pos", "unmapped"}:
        return f"liftover_{row['liftover_status']}"
    if row["liftover_status"] == "mapped_multi_first_used":
        return "liftover_multi_hit_review"
    if not bool(row["chrom_match"]):
        return "chrom_mismatch_after_liftover"
    if not bool(row["pos_match"]):
        return "pos_mismatch_after_liftover"
    if pd.isna(row["REF"]) or pd.isna(row["ALT"]):
        return "ref_alt_missing"
    return "strict_pass"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/6] 读取 GOF/LOF HGMD2019 CSV ...")
    goflof = pd.read_csv(
        GOFLOF_CSV,
        dtype={"ID": "string", "LABEL": "string", "CHROM": "string", "POS": "Int64", "GENE": "string"},
    )
    required_cols = {"ID", "LABEL", "CHROM", "POS", "GENE"}
    missing_cols = required_cols - set(goflof.columns)
    if missing_cols:
        raise ValueError(f"GOF/LOF CSV 缺少必要列: {sorted(missing_cols)}")

    goflof = goflof.rename(columns={"CHROM": "CHROM_hg19", "POS": "POS_hg19"})
    goflof["CHROM_hg19"] = goflof["CHROM_hg19"].map(normalize_chrom)

    print("[2/6] 加载 hg19ToHg38 chain 并执行 liftover ...")
    lo = LiftOver(CHAIN_FILE)
    lifted = goflof.apply(lambda row: first_unique_liftover(lo, row["CHROM_hg19"], row["POS_hg19"]), axis=1)
    lifted_df = pd.DataFrame(list(lifted))
    goflof = pd.concat([goflof, lifted_df], axis=1)

    print("[3/6] 读取 HGMDpro202301 hg38 VCF 风格表 ...")
    hgmd = pd.read_excel(
        HGMD_XLSX,
        sheet_name="Pubmed_2023Q1.hg38",
        header=5,
        dtype={"#CHROM": "string", "POS": "Int64", "ID": "string", "REF": "string", "ALT": "string", "INFO": "string"},
    )
    hgmd = hgmd[["#CHROM", "POS", "ID", "REF", "ALT", "INFO"]].rename(
        columns={
            "#CHROM": "CHROM_hg38_HGMD2023",
            "POS": "POS_hg38_HGMD2023",
            "INFO": "HGMD2023_INFO",
        }
    )
    hgmd["CHROM_hg38_HGMD2023"] = hgmd["CHROM_hg38_HGMD2023"].map(normalize_chrom)

    if hgmd["ID"].duplicated().any():
        dup_count = int(hgmd["ID"].duplicated().sum())
        raise ValueError(f"HGMD2023 ID 存在重复，无法 many-to-one 合并: {dup_count}")

    print("[4/6] 按 HGMD ID 合并并严格比较 CHROM/POS/REF/ALT ...")
    merged = goflof.merge(hgmd, on="ID", how="left", validate="many_to_one")
    merged["ID_found_in_HGMD2023"] = merged["CHROM_hg38_HGMD2023"].notna()
    merged["chrom_match"] = merged["CHROM_hg38_liftover"].astype("string").eq(merged["CHROM_hg38_HGMD2023"].astype("string"))
    merged["pos_match"] = merged["POS_hg38_liftover"].astype("Int64").eq(merged["POS_hg38_HGMD2023"].astype("Int64"))
    merged["ref_alt_available"] = merged["REF"].notna() & merged["ALT"].notna()
    merged["match_status"] = merged.apply(make_match_status, axis=1)
    merged["use_for_project"] = merged["match_status"].eq("strict_pass").map({True: "YES", False: "REVIEW"})
    merged["variant_hg19_original"] = merged["CHROM_hg19"].astype(str) + ":" + merged["POS_hg19"].astype(str)
    merged["variant_hg38_liftover"] = merged["CHROM_hg38_liftover"].astype(str) + ":" + merged["POS_hg38_liftover"].astype(str)
    merged["variant_hg38_HGMD2023"] = (
        merged["CHROM_hg38_HGMD2023"].astype(str)
        + ":"
        + merged["POS_hg38_HGMD2023"].astype(str)
        + ":"
        + merged["REF"].astype(str)
        + ">"
        + merged["ALT"].astype(str)
    )

    output_cols = [
        "ID", "LABEL", "GENE",
        "CHROM_hg19", "POS_hg19", "variant_hg19_original",
        "CHROM_hg38_liftover", "POS_hg38_liftover", "variant_hg38_liftover",
        "CHROM_hg38_HGMD2023", "POS_hg38_HGMD2023", "variant_hg38_HGMD2023",
        "REF", "ALT",
        "liftover_status", "liftover_hits",
        "ID_found_in_HGMD2023", "chrom_match", "pos_match", "ref_alt_available",
        "match_status", "use_for_project", "HGMD2023_INFO",
    ]
    merged = merged[output_cols]

    print("[5/6] 写出完整表、严格通过表、复核表 ...")
    strict_pass = merged[merged["match_status"] == "strict_pass"].copy()
    needs_review = merged[merged["match_status"] != "strict_pass"].copy()

    merged.to_csv(FULL_OUT, index=False, encoding="utf-8-sig")
    strict_pass.to_csv(STRICT_OUT, index=False, encoding="utf-8-sig")
    needs_review.to_csv(REVIEW_OUT, index=False, encoding="utf-8-sig")

    print("[6/6] 写出汇总统计 ...")
    status_counts = Counter(merged["match_status"].tolist())
    label_counts = merged.groupby(["LABEL", "match_status"], dropna=False).size().reset_index(name="count")

    summary_rows = [
        {"metric": "input_rows", "value": len(merged)},
        {"metric": "unique_input_ids", "value": merged["ID"].nunique()},
        {"metric": "liftover_mapped_rows", "value": int(merged["liftover_status"].str.startswith("mapped").sum())},
        {"metric": "liftover_unmapped_or_invalid_rows", "value": int(merged["liftover_status"].isin(["unmapped", "invalid_pos"]).sum())},
        {"metric": "id_found_in_HGMD2023_rows", "value": int(merged["ID_found_in_HGMD2023"].sum())},
        {"metric": "chrom_match_rows", "value": int(merged["chrom_match"].sum())},
        {"metric": "pos_match_rows", "value": int(merged["pos_match"].sum())},
        {"metric": "ref_alt_available_rows", "value": int(merged["ref_alt_available"].sum())},
        {"metric": "strict_pass_rows", "value": len(strict_pass)},
        {"metric": "needs_review_rows", "value": len(needs_review)},
    ]
    for status, count in sorted(status_counts.items()):
        summary_rows.append({"metric": f"match_status:{status}", "value": count})
    for _, row in label_counts.iterrows():
        summary_rows.append({"metric": f"label_status:{row['LABEL']}:{row['match_status']}", "value": int(row["count"])})

    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print("完成。")
    print(f"FULL: {FULL_OUT}")
    print(f"STRICT_PASS: {STRICT_OUT}")
    print(f"NEEDS_REVIEW: {REVIEW_OUT}")
    print(f"SUMMARY: {SUMMARY_OUT}")
    print(pd.DataFrame(summary_rows).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
