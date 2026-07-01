#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 09b 步：为 API 推理链生成抽样输入。

目标：
  1. 控制 API 成本，不对全量 09 prompt 调用模型。
  2. 构建平衡训练集：GOF 2500、LOF 2500、Neutral/Benign 5000。
  3. 优先选择同时具有 GOF 和 LOF 记录的 shared genes。
  4. 尽量限制单个基因贡献的样本数，降低模型记住基因名的风险。

输入：
  09_BioReason_protein_Stage1_Binary.csv
  09_BioReason_protein_Stage2_GOF_LOF.csv

输出：
  09b_BioReason_protein_Stage1_Binary_sampled.csv
  09b_BioReason_protein_Stage2_GOF_LOF_sampled.csv
  09b_reasoning_sample_summary.csv
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Iterable

import pandas as pd


BASE_DIR = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed"
STAGE1_IN = os.path.join(BASE_DIR, "09_BioReason_protein_Stage1_Binary.csv")
STAGE2_IN = os.path.join(BASE_DIR, "09_BioReason_protein_Stage2_GOF_LOF.csv")
STAGE1_OUT = os.path.join(BASE_DIR, "09b_BioReason_protein_Stage1_Binary_sampled.csv")
STAGE2_OUT = os.path.join(BASE_DIR, "09b_BioReason_protein_Stage2_GOF_LOF_sampled.csv")
SUMMARY_OUT = os.path.join(BASE_DIR, "09b_reasoning_sample_summary.csv")

GOF_LABEL = "Gain-of-Function (GOF)"
LOF_LABEL = "Loss-of-Function (LOF)"
PATHOGENIC_LABEL = "Pathogenic"
BENIGN_LABELS = {"Benign", "Neutral", "Benign; Neutral"}


parser = argparse.ArgumentParser(description="第 09b 步：抽样 API 推理链输入")
parser.add_argument("--gof", type=int, default=2500, help="Stage2 GOF 目标条数")
parser.add_argument("--lof", type=int, default=2500, help="Stage2 LOF 目标条数")
parser.add_argument("--neutral", type=int, default=5000, help="Stage1 Neutral/Benign 目标条数")
parser.add_argument("--per_gene_label_cap", type=int, default=30,
                    help="每个基因在每个标签下的软上限；不足时会自动放宽")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()


def parse_gene(question: str) -> str:
    match = re.search(r"^- Gene:\s*(.+?)\s*$", str(question), flags=re.MULTILINE)
    if not match:
        return "Unknown"
    return match.group(1).strip()


def parse_consequence(question: str) -> str:
    match = re.search(r"^- Consequence:\s*(.+?)\s*$", str(question), flags=re.MULTILINE)
    if not match:
        return "Unknown"
    return match.group(1).strip()


def add_sampling_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_gene"] = df["question"].map(parse_gene)
    df["_consequence"] = df["question"].map(parse_consequence)
    return df


def shuffled(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def round_robin_by_gene(
    df: pd.DataFrame,
    target_n: int,
    priority_genes: Iterable[str],
    per_gene_cap: int,
    seed: int,
) -> pd.DataFrame:
    """按基因轮转抽样，先尽量抽 priority_genes，再用其他基因补足。

    这样比简单随机更适合本任务：能提高基因覆盖度，避免少数大基因贡献过多样本。
    per_gene_cap 是软上限；如果目标数不足，会逐轮放宽上限。
    """
    if len(df) <= target_n:
        return shuffled(df, seed)

    priority = {str(g) for g in priority_genes if str(g)}
    work = shuffled(df, seed)

    def build_gene_index(sub_df: pd.DataFrame) -> dict[str, list[int]]:
        gene_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, row in sub_df.iterrows():
            gene_to_indices[str(row["_gene"])].append(idx)
        return gene_to_indices

    def consume_balanced(sub_df: pd.DataFrame, need: int, initial_cap: int) -> list[int]:
        gene_to_indices = build_gene_index(sub_df)
        gene_order = sorted(gene_to_indices)
        selected: list[int] = []
        selected_set: set[int] = set()
        selected_count_by_gene: dict[str, int] = defaultdict(int)
        cap = max(1, initial_cap)

        changed = True
        while len(selected) < need:
            before = len(selected)
            changed = False
            for gene in gene_order:
                if selected_count_by_gene[gene] >= cap:
                    continue
                pool = gene_to_indices[gene]
                while pool and pool[0] in selected_set:
                    pool.pop(0)
                if not pool:
                    continue
                idx = pool.pop(0)
                selected.append(idx)
                selected_set.add(idx)
                selected_count_by_gene[gene] += 1
                changed = True
                if len(selected) >= need:
                    break
            if len(selected) >= need:
                break
            if not changed or len(selected) == before:
                remaining = sub_df[~sub_df.index.isin(selected_set)]
                if len(remaining) == 0:
                    break
                cap *= 2

        return selected

    priority_df = work[work["_gene"].astype(str).isin(priority)].copy()
    fallback_df = work[~work.index.isin(priority_df.index)].copy()

    priority_indices = consume_balanced(priority_df, target_n, per_gene_cap)
    selected = list(priority_indices)

    if len(selected) < target_n:
        fallback_indices = consume_balanced(fallback_df, target_n - len(selected), per_gene_cap)
        selected.extend(fallback_indices)

    sampled = work.loc[selected].copy()

    return shuffled(sampled, seed)


def summarize(name: str, df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"dataset": name, "metric": "rows", "value": int(len(df))},
        {"dataset": name, "metric": "genes", "value": int(df["_gene"].nunique()) if "_gene" in df else 0},
    ]
    if "answer" in df:
        for label, count in df["answer"].value_counts().sort_index().items():
            rows.append({"dataset": name, "metric": f"answer:{label}", "value": int(count)})
    if "_consequence" in df:
        top_cons = df["_consequence"].value_counts().head(10)
        for cons, count in top_cons.items():
            rows.append({"dataset": name, "metric": f"top_consequence:{cons}", "value": int(count)})
    return rows


def main() -> None:
    print("读取 09 阶段全量 prompt ...", flush=True)
    stage1 = add_sampling_columns(pd.read_csv(STAGE1_IN))
    stage2 = add_sampling_columns(pd.read_csv(STAGE2_IN))

    gof_pool = stage2[stage2["answer"].eq(GOF_LABEL)].copy()
    lof_pool = stage2[stage2["answer"].eq(LOF_LABEL)].copy()

    gof_genes = set(gof_pool["_gene"].dropna().astype(str))
    lof_genes = set(lof_pool["_gene"].dropna().astype(str))
    shared_genes = gof_genes & lof_genes

    print(f"Stage2 GOF pool: {len(gof_pool)} 条，{len(gof_genes)} 个基因", flush=True)
    print(f"Stage2 LOF pool: {len(lof_pool)} 条，{len(lof_genes)} 个基因", flush=True)
    print(f"shared GOF/LOF genes: {len(shared_genes)} 个", flush=True)

    gof_sample = round_robin_by_gene(
        gof_pool,
        target_n=args.gof,
        priority_genes=shared_genes,
        per_gene_cap=args.per_gene_label_cap,
        seed=args.seed,
    )
    selected_gof_genes = set(gof_sample["_gene"].dropna().astype(str))
    shared_selected_genes = selected_gof_genes & shared_genes

    lof_sample = round_robin_by_gene(
        lof_pool,
        target_n=args.lof,
        priority_genes=shared_selected_genes or shared_genes,
        per_gene_cap=args.per_gene_label_cap,
        seed=args.seed + 1,
    )

    stage2_sample = shuffled(pd.concat([gof_sample, lof_sample], ignore_index=True), args.seed)

    selected_pathogenic_ids = set(stage2_sample["ID"].astype(str))
    stage1_pathogenic = stage1[
        stage1["ID"].astype(str).str.replace("Task_Stage1_", "Task_Stage2_", regex=False).isin(selected_pathogenic_ids)
    ].copy()

    if len(stage1_pathogenic) != len(stage2_sample):
        # ID 在两个阶段都保留了 07 主表原始索引；正常情况下应完全对应。
        print(f"警告：Stage1 pathogenic 对齐数量 {len(stage1_pathogenic)} != Stage2 sampled {len(stage2_sample)}", flush=True)

    neutral_pool = stage1[stage1["answer"].isin(BENIGN_LABELS)].copy()
    # Neutral 先优先 shared GOF/LOF genes，不足再从其他基因补足。
    priority_neutral_genes = shared_genes
    neutral_sample = round_robin_by_gene(
        neutral_pool,
        target_n=args.neutral,
        priority_genes=priority_neutral_genes,
        per_gene_cap=args.per_gene_label_cap,
        seed=args.seed + 2,
    )

    stage1_sample = shuffled(pd.concat([stage1_pathogenic, neutral_sample], ignore_index=True), args.seed)

    # _gene/_consequence 只用于本脚本抽样和 summary 审计，不写入给第 10 步/API 使用的最终表。
    os.makedirs(BASE_DIR, exist_ok=True)
    output_stage1 = stage1_sample.drop(columns=["_gene", "_consequence"], errors="ignore")
    output_stage2 = stage2_sample.drop(columns=["_gene", "_consequence"], errors="ignore")
    output_stage1.to_csv(STAGE1_OUT, index=False)
    output_stage2.to_csv(STAGE2_OUT, index=False)

    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(summarize("stage1_sample", stage1_sample))
    summary_rows.extend(summarize("stage2_sample", stage2_sample))
    summary_rows.append({"dataset": "sampling", "metric": "shared_gof_lof_genes_total", "value": len(shared_genes)})
    summary_rows.append({"dataset": "sampling", "metric": "stage2_sample_rows_from_shared_genes", "value": int(stage2_sample["_gene"].isin(shared_genes).sum())})
    summary_rows.append({"dataset": "sampling", "metric": "stage2_sample_unique_shared_genes", "value": int(stage2_sample[stage2_sample["_gene"].isin(shared_genes)]["_gene"].nunique())})
    summary_rows.append({"dataset": "sampling", "metric": "neutral_rows_from_shared_genes", "value": int(neutral_sample["_gene"].isin(shared_genes).sum())})
    summary_rows.append({"dataset": "sampling", "metric": "neutral_unique_shared_genes", "value": int(neutral_sample[neutral_sample["_gene"].isin(shared_genes)]["_gene"].nunique())})
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)

    print("抽样完成。", flush=True)
    print(f"Stage1 sampled: {STAGE1_OUT} ({len(stage1_sample)} rows)", flush=True)
    print(stage1_sample["answer"].value_counts().to_string(), flush=True)
    print(f"Stage2 sampled: {STAGE2_OUT} ({len(stage2_sample)} rows)", flush=True)
    print(stage2_sample["answer"].value_counts().to_string(), flush=True)
    print(f"Summary: {SUMMARY_OUT}", flush=True)


if __name__ == "__main__":
    main()
