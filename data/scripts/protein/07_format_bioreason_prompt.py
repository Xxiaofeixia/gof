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

CORE_FEATURES = {
    1: {
        "Consequence",
        "Amino_acids",
        "Protein_position",
        "MAX_AF",
        "AF",
        "CADD_phred",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "AlphaMissense_score",
        "MutPred_score",
        "SpliceAI_DS_max",
        "DOMAINS",
        "Functional_Site",
        "Secondary_Structure",
        "AlphaFold_RSA",
        "ESM_DDG_Score",
        "Spatial_Density_10A",
    },
    2: {
        "Consequence",
        "Amino_acids",
        "Protein_position",
        "SpliceAI_DS_max",
        "in_last_exon",
        "NMD_predicted",
        "protein_truncation_percent",
        "DOMAINS",
        "Functional_Site",
        "Secondary_Structure",
        "AlphaFold_RSA",
        "Spatial_Density_10A",
        "ESM_DDG_Score",
        "Isoelectric_diff",
        "Molecular_weight",
        "pLI",
        "Haploinsufficiency_Score",
        "Inheritance_Pattern",
        "Gene_Normal_Function",
    },
}

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV  = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/06_BIOREASON_with_Biochem.csv"
FEATURE_JSON = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/feature_selection_result.json"
GENE_FUNC_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/gene_function_map.csv"

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

# 读取基因功能映射表 (来自基因流水线 06_fetch_gene_function.py)
gene_func_map = {}
try:
    df_gf = pd.read_csv(GENE_FUNC_CSV)
    for _, row in df_gf.iterrows():
        sym = str(row["SYMBOL"]).strip().upper()
        func = str(row["Gene_Normal_Function"]).strip()
        if func and func != "nan" and func != "Function unknown":
            gene_func_map[sym] = func
    print(f"📋 基因功能映射: {len(gene_func_map)} 条")
except FileNotFoundError:
    print(f"⚠️  未找到 {GENE_FUNC_CSV}，跳过基因功能上下文")

# ==========================================
# 3. 快捷判断函数
# ==========================================
def include(feature_name):
    """判断某个特征是否应该出现在当前阶段的 prompt 中"""
    if feature_name in CORE_FEATURES.get(STAGE, set()):
        return True
    if sig_features is None:
        return True  # 无 JSON 时全部包含
    return feature_name in sig_features

def safe_metric(val):
    """将缺失值统一转为 'Unknown'，包括 NaN、空字符串、占位符文本。"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Unknown"
    v = str(val).strip()
    if v in ("", "-", "nan", "None", "none", "nan"):
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

def is_standard_missense_change(row):
    """只有标准 20 种氨基酸之间的一对一替换，才展示折叠后结构/生化特征。"""
    consequence_text = str(row.get("Consequence", ""))
    if "missense_variant" not in consequence_text:
        return False
    aa_str = str(row.get("Amino_acids", "")).strip()
    if "/" not in aa_str:
        return False
    wt, mt = aa_str.split("/")[0].strip(), aa_str.split("/")[-1].strip()
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    return len(wt) == 1 and len(mt) == 1 and wt in standard and mt in standard

def row_gene_function(row, gene):
    """优先使用 06b 写入主表的基因功能；没有时回退到映射表。"""
    func = safe_metric(row.get("Gene_Normal_Function", ""))
    if func not in ("Unknown", "Function unknown"):
        return func
    return gene_func_map.get(str(gene).strip().upper(), "")

def suggests_absent_or_truncated_product(nmd_text):
    """LOFTEE 解释提示转录本/蛋白产物可能缺失或早期截短时，跳过折叠后特征。"""
    text = str(nmd_text).strip()
    return text.startswith("Likely_NMD_or_early_truncation")

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

    if include("Protein_position"):
        pp = safe_metric(row.get("Protein_position", ""))
        if pp != "Unknown":
            prompt_lines.append(f"- Protein position: {pp}")

    if include("in_last_exon"):
        in_last = safe_metric(row.get("in_last_exon"))
        if in_last != "Unknown":
            prompt_lines.append(f"- In last exon: {'Yes' if int(float(in_last)) == 1 else 'No'}")

    nmd_pred = safe_metric(row.get("NMD_predicted", ""))
    trunc_pct = safe_metric(row.get("protein_truncation_percent", ""))
    use_post_translational_features = (
        is_standard_missense_change(row)
        and not suggests_absent_or_truncated_product(nmd_pred)
    )
    if nmd_pred != "Unknown" or trunc_pct != "Unknown":
        prompt_lines.append("")
        prompt_lines.append("## Transcript and Translation Consequence")
        if nmd_pred != "Unknown":
            prompt_lines.append(f"- LOFTEE NMD/LoF interpretation: {nmd_pred}")
        if trunc_pct != "Unknown":
            prompt_lines.append(f"- Estimated protein truncation percent: {trunc_pct}%")

    # --- 基因正常功能 (基因流水线 06) ---
    gene_func = row_gene_function(row, gene)
    if gene_func:
        prompt_lines.append("")
        prompt_lines.append("## Gene Normal Function")
        prompt_lines.append(f"- {gene_func}")

    # --- 人群频率 (MAX_AF) ---
    if include("MAX_AF"):
        max_af = safe_metric(row.get("MAX_AF", row.get("AF")))
        if max_af != "Unknown":
            prompt_lines.append("")
            prompt_lines.append("## Population & Evolutionary Conservation")
            prompt_lines.append(f"- Maximum allele frequency: {max_af}")

    # --- 保守性 ---
    conservation_lines = []
    if include("phyloP100way_vertebrate"):
        phylop = safe_metric(row.get("phyloP100way_vertebrate"))
        if phylop != "Unknown":
            conservation_lines.append(f"- phyloP 100way: {phylop}")

    if include("GERP++_RS"):
        gerp = safe_metric(row.get("GERP++_RS"))
        if gerp != "Unknown":
            conservation_lines.append(f"- GERP++ RS: {gerp}")

    if conservation_lines:
        prompt_lines.append("")
        prompt_lines.append("## Evolutionary Conservation")
        prompt_lines.extend(conservation_lines)

    # --- 计算致病性预测 ---
    path_pred_lines = []
    if include("AlphaMissense_score"):
        am = safe_metric(row.get("AlphaMissense_score"))
        if am != "Unknown":
            path_pred_lines.append(f"- AlphaMissense: {am}")

    if include("CADD_phred"):
        cadd = safe_metric(row.get("CADD_phred"))
        if cadd != "Unknown":
            path_pred_lines.append(f"- CADD phred: {cadd}")

    if include("MutPred_score"):
        mp = safe_metric(row.get("MutPred_score"))
        if mp != "Unknown":
            path_pred_lines.append(f"- MutPred score: {mp}")

    if path_pred_lines:
        prompt_lines.append("")
        prompt_lines.append("## Computational Pathogenicity Prediction")
        prompt_lines.extend(path_pred_lines)

    # --- 剪接预测 ---
    if include("SpliceAI_DS_max"):
        spliceai = safe_metric(row.get("SpliceAI_DS_max"))
        if spliceai != "Unknown":
            prompt_lines.append("")
            prompt_lines.append("## Splicing Prediction")
            prompt_lines.append(f"- SpliceAI max delta score: {spliceai}")

    # --- 基因级别约束 ---
    gene_constraint_lines = []
    if include("pLI"):
        pli = safe_metric(row.get("pLI"))
        if pli != "Unknown":
            gene_constraint_lines.append(f"- pLI: {pli}")

    if include("Inheritance_Pattern"):
        moi = safe_metric(row.get("Inheritance_Pattern", "Unknown"))
        if moi != "Unknown":
            gene_constraint_lines.append(f"- Inheritance pattern: {moi}")

    if include("Haploinsufficiency_Score"):
        gene_ess = safe_metric(row.get("Haploinsufficiency_Score"))
        if gene_ess != "Unknown":
            gene_constraint_lines.append(f"- Haploinsufficiency score (oe_lof_upper): {gene_ess}")

    if gene_constraint_lines:
        prompt_lines.append("")
        prompt_lines.append("## Gene-Level Constraints")
        prompt_lines.extend(gene_constraint_lines)

    # --- 蛋白质 3D 结构 & 热力学 ---
    has_structure = False
    rsa = str(row.get("AlphaFold_RSA", "Unknown"))
    density = safe_metric(row.get("Spatial_Density_10A"))
    esm_raw = row.get("ESM_DDG_Score")
    if esm_raw is None or (isinstance(esm_raw, float) and pd.isna(esm_raw)):
        esm_val = None
    else:
        try:
            esm_val = round(float(esm_raw), 4)
        except (ValueError, TypeError):
            esm_val = None

    any_structure_feature = (
        include("AlphaFold_RSA") or
        include("Spatial_Density_10A") or
        include("ESM_DDG_Score")
    )

    structure_lines = []
    if any_structure_feature and use_post_translational_features:
        if include("AlphaFold_RSA") and rsa != "Unknown":
            structure_lines.append(f"- Relative solvent accessibility (RSA): {rsa}")
        if include("Spatial_Density_10A") and density != "Unknown":
            structure_lines.append(f"- Spatial density (10A neighbors): {density}")
        if include("ESM_DDG_Score") and esm_val is not None:
            structure_lines.append(f"- ESM-2 stability change (ddG): {esm_val}")

    if structure_lines:
        prompt_lines.append("")
        prompt_lines.append("## Protein 3D Structure & Thermodynamics (AlphaFold)")
        prompt_lines.extend(structure_lines)

    # --- 结构域 ---
    if include("DOMAINS"):
        domains = safe_metric(row.get("DOMAINS", ""))
        if domains not in ("Unknown", "None"):
            prompt_lines.append("")
            prompt_lines.append("## Protein Domains")
            prompt_lines.append(f"- UniProt domains: {domains}")

    # --- 功能位点 (UniProt API) ---
    func_site = str(row.get("Functional_Site", ""))
    if func_site and func_site not in (
        "nan", "", "No annotated functional site at this position", "None"
    ):
        prompt_lines.append("")
        prompt_lines.append("## Functional Site Annotation")
        prompt_lines.append(f"- {func_site}")

    # --- 二级结构 ---
    if include("Secondary_Structure") and use_post_translational_features:
        ss = safe_metric(row.get("Secondary_Structure", ""))
        if ss != "Unknown":
            prompt_lines.append("")
            prompt_lines.append("## Secondary Structure")
            prompt_lines.append(f"- Secondary structure: {ss}")

    # --- 生化特征 ---
    biochem_lines = []
    if include("Isoelectric_diff") and use_post_translational_features:
        iso = safe_metric(row.get("Isoelectric_diff"))
        if iso != "Unknown":
            biochem_lines.append(f"- Isoelectric point difference (|ΔpI|): {iso}")

    if include("Molecular_weight") and use_post_translational_features:
        mw = safe_metric(row.get("Molecular_weight"))
        if mw != "Unknown":
            biochem_lines.append(f"- Molecular weight difference (|ΔMW|): {mw}")

    if biochem_lines:
        prompt_lines.append("")
        prompt_lines.append("## Biochemical Properties")
        prompt_lines.extend(biochem_lines)

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
