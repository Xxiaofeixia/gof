#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 09 步：按阶段构建 BioReason 问题 prompt。

设计原则：
  1. Stage1 和 Stage2 使用不同的证据结构。
  2. reasoning 模式按第 10 步 JSON 推理链的 6 步逻辑组织特征。
  3. classification 模式可保留最终分类指令；默认使用 reasoning 模式。
  4. 08 p 值检验只作为 QC 报告，不再决定 prompt 特征是否加入。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd


parser = argparse.ArgumentParser(description="第 09 步：按阶段构建 BioReason prompt")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="1=Pathogenic vs Benign，2=GOF vs LOF")
parser.add_argument("--mode", choices=["reasoning", "classification"], default="reasoning",
                    help="reasoning=给第10步 API 推理链使用；classification=直接分类训练使用")
args = parser.parse_args()

STAGE = args.stage
MODE = args.mode

BASE_DIR = "/gpfs/hpc/home/lijc/mapengtao/gof/data"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
INPUT_CSV = os.path.join(PROCESSED_DIR, "07_BIOREASON_with_ReasoningContext.csv")

if STAGE == 1:
    OUTPUT_CSV = os.path.join(PROCESSED_DIR, "09_BioReason_protein_Stage1_Binary.csv")
else:
    OUTPUT_CSV = os.path.join(PROCESSED_DIR, "09_BioReason_protein_Stage2_GOF_LOF.csv")

NO_SITE_PLACEHOLDER = "No annotated functional site at this position"
NO_EVIDENCE_PLACEHOLDER = "[No usable evidence provided for this evidence group]"

DOMAIN_SOURCE_PRIORITY = [
    "Pfam",
    "SMART",
    "PROSITE_profiles",
    "PROSITE_patterns",
    "CDD",
    "Gene3D",
    "PANTHER",
    "Superfamily",
    "PIRSF",
    "Prints",
    "InterPro",
]
DOMAIN_DROP_PREFIXES = (
    "PDB-ENSP_mappings:",
    "AlphaFold_DB_import:",
)


def safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "-", "None", "none", "nan", "NaN", "Unknown", "unknown", "Function unknown"}:
        return ""
    return text


def safe_float(value: Any, digits: int = 4) -> str:
    text = safe_text(value)
    if not text:
        return ""
    try:
        return str(round(float(text), digits))
    except ValueError:
        return text


def add_line(lines: list[str], name: str, value: Any, digits: int | None = None) -> None:
    text = safe_float(value, digits) if digits is not None else safe_text(value)
    if text:
        lines.append(f"- {name}: {text}")


def add_section(lines: list[str], title: str, entries: list[str]) -> None:
    entries = [entry for entry in entries if safe_text(entry)]
    lines.append("")
    lines.append(f"## {title}")
    lines.extend(entries if entries else [f"- {NO_EVIDENCE_PLACEHOLDER}"])


def clean_domain_annotations(value: Any, max_items: int = 8) -> str:
    text = safe_text(value)
    if not text:
        return ""

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item or item.startswith(DOMAIN_DROP_PREFIXES):
            continue
        source = item.split(":", 1)[0]
        if source not in DOMAIN_SOURCE_PRIORITY:
            continue
        if item in seen:
            continue
        items.append(item)
        seen.add(item)

    def priority_key(item: str) -> tuple[int, str]:
        source = item.split(":", 1)[0]
        try:
            rank = DOMAIN_SOURCE_PRIORITY.index(source)
        except ValueError:
            rank = len(DOMAIN_SOURCE_PRIORITY)
        return rank, item

    items = sorted(items, key=priority_key)
    if not items:
        return ""

    shown = items[:max_items]
    suffix = "" if len(items) <= max_items else f"; +{len(items) - max_items} more"
    return "; ".join(shown) + suffix


def format_consequence(value: Any) -> str:
    return safe_text(value).replace("_", " ")


def format_aa_change(value: Any) -> str:
    aa = safe_text(value)
    if not aa:
        return ""
    if "/" not in aa:
        return aa
    wt, mt = aa.split("/")[0].strip(), aa.split("/")[-1].strip()
    if mt in {"*", "ter", "Ter", "TER"}:
        return f"{wt} to stop codon"
    return f"{wt} to {mt}"


def is_standard_missense(row: pd.Series) -> bool:
    consequence = str(row.get("Consequence", ""))
    aa = safe_text(row.get("Amino_acids"))
    if "missense_variant" not in consequence or "/" not in aa:
        return False
    wt, mt = aa.split("/")[0].strip(), aa.split("/")[-1].strip()
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    return len(wt) == 1 and len(mt) == 1 and wt in standard and mt in standard


def likely_absent_or_early_truncated(row: pd.Series) -> bool:
    nmd = safe_text(row.get("NMD_predicted"))
    return nmd.startswith("Likely_NMD_or_early_truncation")


def functional_site_text(row: pd.Series) -> str:
    site = safe_text(row.get("Functional_Site"))
    if site in {"", NO_SITE_PLACEHOLDER}:
        return ""
    return site


def in_last_exon_text(row: pd.Series) -> str:
    value = safe_text(row.get("in_last_exon"))
    if not value:
        return ""
    try:
        return "Yes" if int(float(value)) == 1 else "No"
    except ValueError:
        return value


def stage1_question(row: pd.Series) -> str:
    lines = [
        "Predict whether the following genetic variant is Pathogenic or Benign.",
    ]

    step1: list[str] = []
    add_line(step1, "Gene", row.get("SYMBOL"))
    add_line(step1, "Location", row.get("Location"))
    add_line(step1, "REF", row.get("REF"))
    add_line(step1, "ALT", row.get("ALT"))
    consequence = format_consequence(row.get("Consequence"))
    if consequence:
        step1.append(f"- Consequence: {consequence}")
    aa = format_aa_change(row.get("Amino_acids"))
    if aa:
        step1.append(f"- Amino acid change: {aa}")
    add_line(step1, "Protein position", row.get("Protein_position"))
    in_last = in_last_exon_text(row)
    if in_last:
        step1.append(f"- In last exon: {in_last}")
    add_section(lines, "Evidence group 1: Variant consequence", step1)

    step2: list[str] = []
    add_line(step2, "Allele frequency", row.get("AF"), digits=6)
    add_line(step2, "Maximum allele frequency", row.get("MAX_AF"), digits=6)
    add_section(lines, "Evidence group 2: Population tolerance", step2)

    step3: list[str] = []
    add_line(step3, "GERP++ RS", row.get("GERP++_RS"), digits=4)
    add_line(step3, "phyloP 100way vertebrate", row.get("phyloP100way_vertebrate"), digits=4)
    add_section(lines, "Evidence group 3: Evolutionary constraint", step3)

    step4: list[str] = []
    add_line(step4, "AlphaMissense score", row.get("AlphaMissense_score"), digits=4)
    add_line(step4, "MutPred score", row.get("MutPred_score"), digits=4)
    add_line(step4, "CADD phred", row.get("CADD_phred"), digits=4)
    add_section(lines, "Evidence group 4: Molecular damage prediction", step4)

    step5: list[str] = []
    domains = clean_domain_annotations(row.get("DOMAINS"))
    if domains:
        step5.append(f"- Protein domain annotations: {domains}")
    site = functional_site_text(row)
    if site:
        step5.append(f"- Functional site: {site}")
    add_line(step5, "Secondary structure", row.get("Secondary_Structure"))
    if is_standard_missense(row) and not likely_absent_or_early_truncated(row):
        add_line(step5, "AlphaFold RSA", row.get("AlphaFold_RSA"), digits=4)
        add_line(step5, "ESM-2 stability change ddG", row.get("ESM_DDG_Score"), digits=4)
        add_line(step5, "Spatial density within 10A", row.get("Spatial_Density_10A"), digits=4)
    add_line(step5, "pLI", row.get("pLI"), digits=4)
    add_line(step5, "Haploinsufficiency score oe_lof_upper", row.get("Haploinsufficiency_Score"), digits=4)
    add_section(lines, "Evidence group 5: Functional, structural, and gene-level context", step5)

    if MODE == "classification":
        lines.append("")
        lines.append("Output only the classification label: Pathogenic or Benign.")

    return "\n".join(lines)


def stage2_question(row: pd.Series) -> str:
    lines = [
        "Predict whether the following pathogenic genetic variant acts through Gain-of-Function (GOF) or Loss-of-Function (LOF).",
    ]

    step1: list[str] = []
    add_line(step1, "Gene", row.get("SYMBOL"))
    add_line(step1, "Location", row.get("Location"))
    consequence = format_consequence(row.get("Consequence"))
    if consequence:
        step1.append(f"- Consequence: {consequence}")
    aa = format_aa_change(row.get("Amino_acids"))
    if aa:
        step1.append(f"- Amino acid change: {aa}")
    add_line(step1, "Protein position", row.get("Protein_position"))
    add_section(lines, "Evidence group 1: Initial molecular consequence", step1)

    step2: list[str] = []
    if consequence:
        step2.append(f"- Consequence: {consequence}")
    in_last = in_last_exon_text(row)
    if in_last:
        step2.append(f"- In last exon: {in_last}")
    add_line(step2, "LOFTEE NMD/LoF interpretation", row.get("NMD_predicted"))
    add_line(step2, "Estimated protein lost percent", row.get("protein_truncation_percent"), digits=2)
    add_section(lines, "Evidence group 2: Transcript and translation consequence", step2)

    step3: list[str] = []
    domains = clean_domain_annotations(row.get("DOMAINS"))
    if domains:
        step3.append(f"- Protein domain annotations: {domains}")
    site = functional_site_text(row)
    if site:
        step3.append(f"- Functional site: {site}")
    add_line(step3, "Protein position", row.get("Protein_position"))
    add_line(step3, "Secondary structure", row.get("Secondary_Structure"))
    add_line(step3, "AlphaFold RSA", row.get("AlphaFold_RSA"), digits=4)
    add_line(step3, "Spatial density within 10A", row.get("Spatial_Density_10A"), digits=4)
    add_section(lines, "Evidence group 3: Functional region context", step3)

    step4: list[str] = []
    if is_standard_missense(row) and not likely_absent_or_early_truncated(row):
        add_line(step4, "ESM-2 stability change ddG", row.get("ESM_DDG_Score"), digits=4)
        add_line(step4, "AlphaFold RSA", row.get("AlphaFold_RSA"), digits=4)
        add_line(step4, "Isoelectric point difference abs_delta_pI", row.get("Isoelectric_diff"), digits=4)
        add_line(step4, "Molecular weight difference abs_delta_MW", row.get("Molecular_weight"), digits=4)
    add_section(lines, "Evidence group 4: Post-translational biophysical effect", step4)

    step5: list[str] = []
    add_line(step5, "Gene normal function", row.get("Gene_Normal_Function"))
    add_line(step5, "pLI", row.get("pLI"), digits=4)
    add_line(step5, "Haploinsufficiency score oe_lof_upper", row.get("Haploinsufficiency_Score"), digits=4)
    add_line(step5, "Inheritance pattern", row.get("Inheritance_Pattern"))
    add_section(lines, "Evidence group 5: Gene function and pathway context", step5)

    if MODE == "classification":
        lines.append("")
        lines.append("Output only the classification label: Gain-of-Function (GOF) or Loss-of-Function (LOF).")

    return "\n".join(lines)


def answer_from_label(label: str) -> str:
    label = str(label).strip().upper()
    if STAGE == 1:
        return "Pathogenic" if label in {"GOF", "LOF"} else "Benign"
    if label == "GOF":
        return "Gain-of-Function (GOF)"
    if label == "LOF":
        return "Loss-of-Function (LOF)"
    return "Unknown"


def main() -> None:
    print(f"读取特征表: {INPUT_CSV}", flush=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    if STAGE == 2:
        labels = df["LABEL"].astype(str).str.strip().str.upper()
        df = df[labels.isin(["GOF", "LOF"])].copy()
        print(f"Stage2 仅保留 GOF/LOF: {len(df)} 条", flush=True)

    data = {
        "ID": [],
        "question": [],
        "answer": [],
        "reference_sequence": [],
        "variant_sequence": [],
        "gene_type": [],
    }

    for idx, row in df.iterrows():
        data["ID"].append(f"Task_Stage{STAGE}_{idx}")
        data["question"].append(stage1_question(row) if STAGE == 1 else stage2_question(row))
        data["answer"].append(answer_from_label(row.get("LABEL")))
        data["reference_sequence"].append(str(row.get("reference_sequence", "N")))
        data["variant_sequence"].append(str(row.get("variant_sequence", "N")))
        data["gene_type"].append(str(row.get("GENE_TYPE", "shared")))

    out = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("=" * 70, flush=True)
    print(f"第 09 步完成: stage={STAGE}, mode={MODE}", flush=True)
    print(f"样本数: {len(out)}", flush=True)
    print(out["answer"].value_counts().to_string(), flush=True)
    print(f"输出: {OUTPUT_CSV}", flush=True)
    print("=" * 70, flush=True)
    if len(out):
        print(out["question"].iloc[0], flush=True)


if __name__ == "__main__":
    main()
