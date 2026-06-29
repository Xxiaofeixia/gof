"""
自动特征筛选脚本 — 两阶段 p 值检验
====================================

目的: 替代人工看 p 值 → 硬编码删特征的流程，自动化统计检验。

两阶段:
  阶段一 (Stage 1): Pathogenic (GOF+LOF) vs Neutral — 致病性二分类
  阶段二 (Stage 2): GOF vs LOF — 效应机制二分类

检验方法:
  - 连续型特征: Mann-Whitney U test (非参数，不假设正态分布)
  - 分类型特征: Chi-squared test

输出:
  feature_selection_result.json — 每个阶段哪些特征 p<0.05
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ==========================================
# 配置
# ==========================================
INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/06_BIOREASON_with_Biochem.csv"
OUTPUT_JSON = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/feature_selection_result.json"

# 要从 CSV 中检验的特征列及其类型
#   "continuous": Mann-Whitney U
#   "categorical": Chi-squared
FEATURE_CONFIG = {
    "MAX_AF":                    "continuous",
    "AlphaMissense_score":       "continuous",
    "CADD_phred":                "continuous",
    "GERP++_RS":                 "continuous",
    "phyloP100way_vertebrate":   "continuous",
    "pLI":                       "continuous",
    "Haploinsufficiency_Score":  "continuous",
    "AlphaFold_RSA":             "continuous",
    "ESM_DDG_Score":             "continuous",
    "Spatial_Density_10A":       "continuous",
    "Protein_position":          "continuous",
    "SpliceAI_DS_max":           "continuous",
    "MutPred_score":             "continuous",
    "Isoelectric_diff":          "continuous",
    "Molecular_weight":          "continuous",
    "protein_truncation_percent": "continuous",
    "in_last_exon":              "categorical",
    "NMD_predicted":             "categorical",
    "Functional_Site_present":   "categorical",
    # 分类型特征
    "Consequence":               "categorical",
    "Inheritance_Pattern":       "categorical",
    "DOMAINS":                   "categorical",
    "Secondary_Structure":       "categorical",
}

P_THRESHOLD = 0.05
NO_SITE_PLACEHOLDER = "No annotated functional site at this position"


def safe_numeric(series):
    """转为数值，非数值和 NaN 变为 NaN"""
    return pd.to_numeric(series, errors="coerce")


def format_pvalue(value):
    """打印 p 值；无法检验时显示 NA。"""
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4g}"


def add_derived_features(df):
    """为统计检验派生低维、可检验特征。"""
    if "Functional_Site" in df.columns:
        site = df["Functional_Site"].fillna(NO_SITE_PLACEHOLDER).astype(str).str.strip()
        df["Functional_Site_present"] = np.where(
            site.isin(["", "nan", "None", NO_SITE_PLACEHOLDER]),
            "No",
            "Yes",
        )
    return df


def test_continuous(data_df, feature, group_a_mask, group_b_mask, group_a_name, group_b_name):
    """
    Mann-Whitney U 检验。

    返回: {feature, p_value, significant, median_A, median_B, n_A, n_B, note}
    """
    vals = safe_numeric(data_df[feature])
    a = vals[group_a_mask].dropna()
    b = vals[group_b_mask].dropna()

    if len(a) < 5 or len(b) < 5:
        return {
            "feature": feature,
            "p_value": None,
            "significant": False,
            "median_A": float(a.median()) if len(a) > 0 else None,
            "median_B": float(b.median()) if len(b) > 0 else None,
            "n_A": len(a), "n_B": len(b),
            "note": "样本量不足 (<5)，跳过检验"
        }

    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception as e:
        return {
            "feature": feature,
            "p_value": None, "significant": False,
            "median_A": float(a.median()), "median_B": float(b.median()),
            "n_A": len(a), "n_B": len(b),
            "note": f"检验失败: {e}"
        }

    return {
        "feature": feature,
        "p_value": float(p),
        "significant": p < P_THRESHOLD,
        "median_A": float(a.median()),
        "median_B": float(b.median()),
        "n_A": len(a), "n_B": len(b),
        "test": "Mann-Whitney U",
        "note": ""
    }


def test_categorical(data_df, feature, group_a_mask, group_b_mask, group_a_name, group_b_name):
    """
    Chi-squared 独立性检验。

    构建 2×K 列联表（行=分组, 列=特征类别），检验两组的类别分布是否有差异。
    """
    a_labels = data_df.loc[group_a_mask, feature].fillna("Unknown").astype(str)
    b_labels = data_df.loc[group_b_mask, feature].fillna("Unknown").astype(str)

    all_cats = sorted(set(a_labels.unique()) | set(b_labels.unique()))
    if len(all_cats) < 2:
        return {
            "feature": feature, "p_value": None, "significant": False,
            "categories": all_cats,
            "note": "类别数 <2，无法检验"
        }

    # 构建列联表
    table = []
    for grp_vals in [a_labels, b_labels]:
        row = [int((grp_vals == cat).sum()) for cat in all_cats]
        table.append(row)
    table = np.array(table)

    # 检查是否有全零列
    if (table.sum(axis=0) == 0).any():
        return {
            "feature": feature, "p_value": None, "significant": False,
            "categories": all_cats,
            "note": "存在全零列，无法检验"
        }

    try:
        chi2, p, dof, expected = stats.chi2_contingency(table)
    except Exception as e:
        return {
            "feature": feature, "p_value": None, "significant": False,
            "categories": all_cats,
            "note": f"Chi-squared 失败: {e}"
        }

    # 计算每组各类别的占比
    a_props = {str(cat): round(table[0][i] / len(a_labels), 3) for i, cat in enumerate(all_cats)}
    b_props = {str(cat): round(table[1][i] / len(b_labels), 3) for i, cat in enumerate(all_cats)}

    return {
        "feature": feature,
        "p_value": float(p),
        "significant": p < P_THRESHOLD,
        "n_A": len(a_labels), "n_B": len(b_labels),
        "categories": all_cats,
        "group_A_proportions": a_props,
        "group_B_proportions": b_props,
        "test": "Chi-squared",
        "note": ""
    }


def main():
    print("=" * 70)
    print("📊 自动特征筛选 — 两阶段统计检验")
    print("=" * 70)

    # 1. 读数据
    print(f"\n📂 读取: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = add_derived_features(df)
    print(f"   总样本: {len(df)}")

    # 2. 构建分组 mask
    label_col = "LABEL"
    labels = df[label_col].str.strip().str.upper()

    is_gof = (labels == "GOF")
    is_lof = (labels == "LOF")
    is_neutral = (labels == "NEUTRAL") | (labels == "BENIGN")

    # 阶段一: Pathogenic(GOF+LOF) vs Neutral
    is_pathogenic = is_gof | is_lof

    # 阶段二: GOF vs LOF
    is_gof_or_lof = is_gof | is_lof

    n_path = is_pathogenic.sum()
    n_neutral = is_neutral.sum()
    n_gof = is_gof.sum()
    n_lof = is_lof.sum()

    print(f"\n📊 样本分布:")
    print(f"   GOF:       {n_gof}")
    print(f"   LOF:       {n_lof}")
    print(f"   Neutral:   {n_neutral}")
    print(f"   Pathogenic (GOF+LOF): {n_path}")

    # 3. 逐特征检验
    results = {"stage1": {}, "stage2": {}, "summary": {}}

    for feature, ftype in FEATURE_CONFIG.items():
        if feature not in df.columns:
            print(f"   ⚠️ 跳过 '{feature}': CSV 中不存在")
            continue

        test_fn = test_continuous if ftype == "continuous" else test_categorical

        # ---- 阶段一: Pathogenic vs Neutral ----
        r1 = test_fn(df, feature, is_pathogenic, is_neutral, "Pathogenic", "Neutral")
        results["stage1"][feature] = r1

        # ---- 阶段二: GOF vs LOF ----
        r2 = test_fn(df, feature, is_gof, is_lof, "GOF", "LOF")
        results["stage2"][feature] = r2

    # 4. 汇总
    stage1_sig = [f for f, r in results["stage1"].items() if r["significant"]]
    stage2_sig = [f for f, r in results["stage2"].items() if r["significant"]]
    both_sig = [f for f in stage1_sig if f in stage2_sig]
    only_stage1 = [f for f in stage1_sig if f not in stage2_sig]
    only_stage2 = [f for f in stage2_sig if f not in stage1_sig]

    results["summary"] = {
        "stage1_significant": stage1_sig,
        "stage2_significant": stage2_sig,
        "both_significant": both_sig,
        "only_stage1": only_stage1,
        "only_stage2": only_stage2,
        "p_threshold": P_THRESHOLD,
    }

    # 5. 输出
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # 6. 打印报告
    print(f"\n{'=' * 70}")
    print(f"📋 检验结果 (p < {P_THRESHOLD})")
    print(f"{'=' * 70}")

    print(f"\n🔴 阶段一 (Pathogenic vs Neutral) 显著特征 ({len(stage1_sig)}):")
    for f in stage1_sig:
        r = results["stage1"][f]
        print(f"   ✅ {f}: p={format_pvalue(r['p_value'])}, "
              f"Pathogenic median={r.get('median_A','N/A')}, "
              f"Neutral median={r.get('median_B','N/A')}")

    not_sig_s1 = [f for f in FEATURE_CONFIG if f not in stage1_sig and f in results["stage1"]]
    for f in not_sig_s1:
        r = results["stage1"][f]
        print(f"   ❌ {f}: p={format_pvalue(r['p_value'])} {r.get('note','')}")

    print(f"\n🔵 阶段二 (GOF vs LOF) 显著特征 ({len(stage2_sig)}):")
    for f in stage2_sig:
        r = results["stage2"][f]
        print(f"   ✅ {f}: p={format_pvalue(r['p_value'])}, "
              f"GOF median={r.get('median_A','N/A')}, "
              f"LOF median={r.get('median_B','N/A')}")

    not_sig_s2 = [f for f in FEATURE_CONFIG if f not in stage2_sig and f in results["stage2"]]
    for f in not_sig_s2:
        r = results["stage2"][f]
        print(f"   ❌ {f}: p={format_pvalue(r['p_value'])} {r.get('note','')}")

    print(f"\n📌 只在阶段一显著 ({len(only_stage1)}): {only_stage1 if only_stage1 else '无'}")
    print(f"📌 只在阶段二显著 ({len(only_stage2)}): {only_stage2 if only_stage2 else '无'}")
    print(f"📌 两阶段都显著 ({len(both_sig)}): {both_sig if both_sig else '无'}")

    print(f"\n💾 详细结果写入: {OUTPUT_JSON}")
    print("✅ 完成。")


if __name__ == "__main__":
    main()
