"""
BioReason 多模态 Prompt 封装引擎 (v2 — 两阶段特征自动筛选)
============================================================

v2 改动:
  - 支持 --stage 1 / --stage 2，分别生成两个训练数据集
  - 读取 00_statistical_feature_selection.py 产出的 feature_selection_result.json
  - 只将 p<0.05 的特征写入 prompt，其余自动跳过
  - 阶段一: Pathogenic vs Benign 二分类，全量样本
  - 阶段二: GOF vs LOF 二分类，仅致病样本

用法:
  python 06_format_bioreason_prompt.py --stage 1
  python 06_format_bioreason_prompt.py --stage 2
"""

import argparse
import json
import os
import pandas as pd
import numpy as np

# ==========================================
# 0. 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="BioReason Prompt 封装 — 两阶段特征筛选")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="1=阶段一(致病vs中性), 2=阶段二(GOF vs LOF)")
args = parser.parse_args()
STAGE = args.stage

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV  = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/05_BIOREASON_with_Topology.csv"
FEATURE_JSON = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/feature_selection_result.json"

if STAGE == 1:
    OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage1_Binary.csv"
else:
    OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage2_GOF_LOF.csv"

# ==========================================
# 2. 读取特征筛选结果
# ==========================================
try:
    with open(FEATURE_JSON, "r") as f:
        fs_result = json.load(f)
    stage_key = f"stage{STAGE}"
    sig_features = set(fs_result["summary"][f"stage{STAGE}_significant"])
    print(f"📋 阶段{STAGE}显著特征 (p<0.05): {sig_features}")
except FileNotFoundError:
    print(f"⚠️  未找到 {FEATURE_JSON}，回退为包含全部特征")
    sig_features = None  # 全部特征都包含

# ==========================================
# 3. 快捷判断函数
# ==========================================
def include(feature_name):
    """判断某个特征是否应该出现在当前阶段的 prompt 中"""
    if sig_features is None:
        return True  # 无 JSON 时全部包含
    return feature_name in sig_features

def safe_metric(val):
    v = str(val).strip()
    if pd.isna(val) or v in ("", "-", "nan", "None"):
        return "Unknown"
    return v

def format_aa_change(aa_raw):
    aa_str = str(aa_raw).strip()
    if aa_str in ("", "-", "None", "nan", "unknown"):
        return None
    if "/" not in aa_str:
        return aa_str
    parts = aa_str.split("/")
    wt = parts[0].strip()
    mt = parts[-1].strip()
    if mt in ("*", "ter", "Ter", "TER"):
        return f"{wt} to stop codon"
    return f"{wt} to {mt}"

# ==========================================
# 4. 读取数据
# ==========================================
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"🔍 读取特征表: {len(df)} 条")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# 阶段二只保留致病样本 (GOF + LOF)
if STAGE == 2:
    labels_upper = df["LABEL"].str.strip().str.upper()
    mask_pathogenic = labels_upper.isin(["GOF", "LOF"])
    df = df[mask_pathogenic].copy()
    print(f"✂️  阶段二过滤: 仅保留 GOF+LOF → {len(df)} 条")

# ==========================================
# 5. 数据结构
# ==========================================
bioreason_data = {
    "ID": [],
    "question": [],
    "answer": [],
    "reference_sequence": [],
    "variant_sequence": [],
    "gene_type": [],
}

print(f"⏳ 正在构建阶段{STAGE} prompt (动态特征筛选)...")

# ==========================================
# 6. 逐行封装
# ==========================================
for idx, row in df.iterrows():
    row_idx = idx  # 保留原始索引
    bioreason_data["ID"].append(f"Task_Stage{STAGE}_{row_idx}")

    gene = str(row.get("SYMBOL", "Unknown"))
    loc  = str(row.get("Location", "Unknown"))
    consequence = str(row.get("Consequence", "unknown variant")).replace("_", " ")
    aa_text = format_aa_change(row.get("Amino_acids", ""))

    # --- 阶段相关配置 ---
    if STAGE == 1:
        task_instruction = (
            "Classify this genetic variant as: Pathogenic or Benign."
        )
    else:
        task_instruction = (
            "Classify this pathogenic variant as: "
            "Gain-of-Function (GOF) or Loss-of-Function (LOF)."
        )

    # ---------------------------------------------------------
    # 动态 Prompt 拼装 — 只包含 p<0.05 的特征
    # ---------------------------------------------------------
    prompt_lines = [
        "Predict the functional effect of the following genetic variant. ",
        task_instruction,
        "",
        "## Variant Information",
        f"- Gene: {gene}",
        f"- Location: {loc}",
        f"- Consequence: {consequence}",
    ]
    if aa_text:
        prompt_lines.append(f"- Amino acid change: {aa_text}")

    # --- 人群频率 (MAX_AF) ---
    if include("MAX_AF"):
        max_af = row.get("MAX_AF", row.get("AF", 0.0))
        if pd.isna(max_af):
            max_af = 0.0
        # 如果整个 Population 小节还没开始过就先加标题
        prompt_lines.append("")
        prompt_lines.append("## Population & Evolutionary Conservation")
        prompt_lines.append(f"- Maximum allele frequency: {max      _af}")

    # --- 保守性 ---
    has_conservation = False
    if include("phyloP100way_vertebrate"):
        if not has_conservation:
            prompt_lines.append("")
            prompt_lines.append("## Evolutionary Conservation")
            has_conservation = True
        phylop = safe_metric(row.get("phyloP100way_vertebrate"))
        if phylop != "Unknown":
            prompt_lines.append(f"- phyloP 100way: {phylop}")

    if include("GERP++_RS"):
        if not has_conservation:
            prompt_lines.append("")
            prompt_lines.append("## Evolutionary Conservation")
            has_conservation = True
        gerp = safe_metric(row.get("GERP++_RS"))
        if gerp != "Unknown":
            prompt_lines.append(f"- GERP++ RS: {gerp}")

    # --- 计算致病性预测 ---
    has_path_pred = False
    if include("AlphaMissense_score"):
        has_path_pred = True
        prompt_lines.append("")
        prompt_lines.append("## Computational Pathogenicity Prediction")
        am = safe_metric(row.get("AlphaMissense_score"))
        if am != "Unknown":
            prompt_lines.append(f"- AlphaMissense: {am}")

    if include("CADD_phred"):
        if not has_path_pred:
            prompt_lines.append("")
            prompt_lines.append("## Computational Pathogenicity Prediction")
            has_path_pred = True
        cadd = safe_metric(row.get("CADD_phred"))
        if cadd != "Unknown":
            prompt_lines.append(f"- CADD phred: {cadd}")

    # --- 基因级别约束 ---
    has_gene_constraint = False
    if include("Inheritance_Pattern"):
        has_gene_constraint = True
        prompt_lines.append("")
        prompt_lines.append("## Gene-Level Constraints")
        moi = str(row.get("Inheritance_Pattern", "Unknown"))
        if moi != "Unknown":
            prompt_lines.append(f"- Inheritance pattern: {moi}")

    if include("Haploinsufficiency_Score"):
        if not has_gene_constraint:
            prompt_lines.append("")
            prompt_lines.append("## Gene-Level Constraints")
            has_gene_constraint = True
        gene_ess = safe_metric(row.get("Haploinsufficiency_Score"))
        if gene_ess != "Unknown":
            prompt_lines.append(f"- Haploinsufficiency score (oe_lof_upper): {gene_ess}")

    # --- 蛋白质 3D 结构 & 热力学 ---
    has_structure = False
    rsa = str(row.get("AlphaFold_RSA", "Unknown"))
    density = str(row.get("Spatial_Density_10A", "0"))
    esm_raw = row.get("ESM_DDG_Score", 0.0)
    try:
        esm_val = round(float(esm_raw), 4)
    except (ValueError, TypeError):
        esm_val = None

    any_structure_feature = (
        include("AlphaFold_RSA") or
        include("Spatial_Density_10A") or
        include("ESM_DDG_Score")
    )

    if any_structure_feature:
        prompt_lines.append("")
        prompt_lines.append("## Protein 3D Structure & Thermodynamics (AlphaFold)")

        if rsa == "Unknown" or esm_val is None:
            prompt_lines.append("- Structural features not available for this variant.")
        else:
            if include("AlphaFold_RSA"):
                prompt_lines.append(f"- Relative solvent accessibility (RSA): {rsa}")
            if include("Spatial_Density_10A"):
                prompt_lines.append(f"- Spatial density (10A neighbors): {density}")
            if include("ESM_DDG_Score") and esm_val is not None:
                prompt_lines.append(f"- ESM-2 stability change (ddG): {esm_val}")

    # --- 结构域 ---
    if include("DOMAINS"):
        prompt_lines.append("")
        prompt_lines.append("## Protein Domains")
        domains = safe_metric(row.get("DOMAINS", ""))
        if domains != "Unknown":
            prompt_lines.append(f"- UniProt domains: {domains}")
        else:
            prompt_lines.append("- Domain information not available.")

    # --- 二级结构 ---
    if include("Secondary_Structure"):
        ss = safe_metric(row.get("Secondary_Structure", ""))
        prompt_lines.append(f"- Secondary structure: {ss}")

    # --- 结尾指令 ---
    prompt_lines.append("")
    prompt_lines.append(
        "Based on the features above, output only the classification label."
    )

    question = "\n".join(prompt_lines)
    bioreason_data["question"].append(question)

    # ---------------------------------------------------------
    # 标签 → 阶段特定的答案
    # ---------------------------------------------------------
    label = str(row.get("LABEL", "Unknown")).strip().upper()

    if STAGE == 1:
        # 阶段一: 二分类 — Pathogenic / Benign
        if label in ("GOF", "LOF"):
            ans = "Pathogenic"
        else:
            ans = "Benign"
    else:
        # 阶段二: 二分类 — GOF / LOF
        if label == "GOF":
            ans = "Gain-of-Function (GOF)"
        elif label == "LOF":
            ans = "Loss-of-Function (LOF)"
        else:
            ans = "Unknown"  # 理论上阶段二不会有 Neutral

    bioreason_data["answer"].append(ans)

    # --- DNA 序列 ---
    ref_seq = str(row.get("reference_sequence", "N"))
    var_seq = str(row.get("variant_sequence", "N"))
    bioreason_data["reference_sequence"].append(ref_seq)
    bioreason_data["variant_sequence"].append(var_seq)

    # --- 基因类型 (用于训练时基因感知分割) ---
    gene_type = str(row.get("GENE_TYPE", "shared"))
    bioreason_data["gene_type"].append(gene_type)

# ==========================================
# 7. 导出
# ==========================================
df_bioreason = pd.DataFrame(bioreason_data)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_bioreason.to_csv(OUTPUT_CSV, index=False)

# 统计标签分布
label_counts = df_bioreason["answer"].value_counts()

print("\n" + "=" * 70)
print(f"🎉 阶段{STAGE} LLM 微调数据集已生成！")
print(f"   样本总数: {len(df_bioreason)}")
print(f"   标签分布:")
for lbl, cnt in label_counts.items():
    print(f"     {lbl}: {cnt}")
print(f"💾 保存至: {OUTPUT_CSV}")
print("=" * 70)
print("\n👇 第一条样本的 Prompt 预览:")
print(df_bioreason["question"].iloc[0])
