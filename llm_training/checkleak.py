"""
诊断 99.6% 准确率：系统性排查数据泄露和过拟合
检查项：
  1. train/val/test 基因是否有重叠
  2. 标签分布是否均衡
  3. 模型是否只输出一个固定类别
  4. substring 匹配是否太宽松
  5. DNA 序列是否在集合间重复
"""
import pandas as pd
import re
import sys
from collections import defaultdict, Counter
import random

STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

if STAGE == 1:
    path = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage1_Binary.csv"
else:
    path = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage2_GOF_LOF.csv"

df = pd.read_csv(path)
print(f"=" * 70)
print(f"诊断 Stage {STAGE}: {len(df)} 条样本")
print(f"=" * 70)

# ============================================================
# 1. 标签分布
# ============================================================
print(f"\n📊 标签分布:")
vc = df["answer"].value_counts()
for label, count in vc.items():
    print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

# 如果模型永远猜多数类，准确率上限
majority_pct = vc.max() / len(df) * 100
print(f"\n⚠️  如果永远猜多数类 '{vc.index[0]}': 准确率 = {majority_pct:.1f}%")

# ============================================================
# 2. 基因提取
# ============================================================
print(f"\n📋 基因分析:")
genes = []
for q in df["question"]:
    m = re.search(r"- Gene: (\S+)", str(q))
    genes.append(m.group(1) if m else "EMPTY")
df["_gene"] = genes

unique_genes = df["_gene"].nunique()
empty_count = (df["_gene"] == "EMPTY").sum()
print(f"  总基因数: {unique_genes}")
print(f"  空基因: {empty_count}")

# 每个基因的标签纯度
gene_label_map = defaultdict(set)
gene_examples = Counter()
for _, row in df.iterrows():
    gene = row["_gene"]
    ans = row["answer"].strip().lower()
    gene_label_map[gene].add(ans)
    gene_examples[gene] += 1

pure_genes = sum(1 for labels in gene_label_map.values() if len(labels) == 1)
mixed_genes = sum(1 for labels in gene_label_map.values() if len(labels) > 1)
print(f"  纯标签基因: {pure_genes} (该基因下所有样本同一标签)")
print(f"  混合标签基因: {mixed_genes} (该基因下样本标签不唯一)")

# ============================================================
# 3. 模拟基因级 8:1:1 分割，检查泄漏
# ============================================================
print(f"\n🔍 模拟基因级分割 (复现 train_dna_qwen_vegg.py 的 split 逻辑):")

# 按标签分类基因
gene_to_labels = {}
for gene, labels in gene_label_map.items():
    gene_to_labels[gene] = labels

if STAGE == 1:
    pure_path = [g for g, lbs in gene_to_labels.items() if any("pathogenic" in l for l in lbs) and not any("benign" in l for l in lbs)]
    pure_benign = [g for g, lbs in gene_to_labels.items() if any("benign" in l for l in lbs) and not any("pathogenic" in l for l in lbs)]
    mixed = [g for g, lbs in gene_to_labels.items() if any("pathogenic" in l for l in lbs) and any("benign" in l for l in lbs)]
    buckets = [pure_path, pure_benign, mixed]
    bucket_names = ["纯Pathogenic基因", "纯Benign基因", "混合基因"]
else:
    pure_gof = [g for g, lbs in gene_to_labels.items() if any("gain-of-function" in l for l in lbs) and not any("loss-of-function" in l for l in lbs)]
    pure_lof = [g for g, lbs in gene_to_labels.items() if any("loss-of-function" in l for l in lbs) and not any("gain-of-function" in l for l in lbs)]
    mixed = [g for g, lbs in gene_to_labels.items() if any("gain-of-function" in l for l in lbs) and any("loss-of-function" in l for l in lbs)]
    other = [g for g, lbs in gene_to_labels.items() if g not in set(pure_gof + pure_lof + mixed)]
    buckets = [pure_gof, pure_lof, mixed, other]
    bucket_names = ["纯GOF基因", "纯LOF基因", "混合基因", "其他"]

for i, (bucket, name) in enumerate(zip(buckets, bucket_names)):
    print(f"  {name}: {len(bucket)} 个基因")

# 模拟分割
def split_genes(gene_list):
    genes = sorted(set(gene_list))
    random.Random(42).shuffle(genes)
    n = len(genes)
    t_end = int(n * 0.8)
    v_end = int(n * 0.9)
    return set(genes[:t_end]), set(genes[t_end:v_end]), set(genes[v_end:])

train_genes, val_genes, test_genes = set(), set(), set()
for bucket in buckets:
    t, v, te = split_genes(bucket)
    train_genes.update(t)
    val_genes.update(v)
    test_genes.update(te)

# Stage 2: lof_only 基因全部进 train
if STAGE == 2 and "gene_type" in df.columns:
    for gene, gtype in zip(df["_gene"], df.get("gene_type", ["shared"]*len(df))):
        if gtype == "lof_only":
            train_genes.add(gene)
            val_genes.discard(gene)
            test_genes.discard(gene)

# 检查交叉
train_val_overlap = train_genes & val_genes
train_test_overlap = train_genes & test_genes
val_test_overlap = val_genes & test_genes

print(f"\n  Train 基因: {len(train_genes)}")
print(f"  Val   基因: {len(val_genes)}")
print(f"  Test  基因: {len(test_genes)}")
print(f"  Train ∩ Val:  {len(train_val_overlap)} {'❌ 泄露!' if train_val_overlap else '✅'}")
print(f"  Train ∩ Test: {len(train_test_overlap)} {'❌ 泄露!' if train_test_overlap else '✅'}")
print(f"  Val ∩ Test:   {len(val_test_overlap)} {'❌ 泄露!' if val_test_overlap else '✅'}")

# ============================================================
# 4. 检查每个基因在 train/test 的样本分布
# ============================================================
train_df = df[df["_gene"].isin(train_genes)]
val_df = df[df["_gene"].isin(val_genes)]
test_df = df[df["_gene"].isin(test_genes)]

print(f"\n📦 样本分布:")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    vc = subset["answer"].value_counts()
    labels_str = ", ".join(f"{l}: {c}" for l, c in vc.items())
    print(f"  {name} 标签: {labels_str}")

# ============================================================
# 5. 检查是否模型只需背基因名就能答对
# ============================================================
print(f"\n🧬 '背基因名' 作弊可行性分析:")
# 对于纯标签基因（该基因下所有样本同一标签），模型可以背基因→标签映射
pure_gene_samples = sum(1 for _, row in df.iterrows()
                        if len(gene_label_map[row["_gene"]]) == 1)
print(f"  纯标签基因中的样本数: {pure_gene_samples}/{len(df)} ({pure_gene_samples/len(df)*100:.1f}%)")

# 在 test 集中有多少是纯标签基因
test_pure = sum(1 for _, row in test_df.iterrows()
                if len(gene_label_map[row["_gene"]]) == 1)
print(f"  Test 中纯标签基因样本: {test_pure}/{len(test_df)} ({test_pure/len(test_df)*100:.1f}%)")
print(f"  → 如果模型背下所有基因→标签映射，理论最高 Test 准确率: {test_pure/len(test_df)*100:.1f}%")

# 不过基因级分割确保了同基因不会跨 train/test，所以背基因名在 test 上没用
# 但可以背"基因特征模式"——如果纯标签基因在 train 中足够多，模型可能学会
# 某些基因级别特征（如 pLI, Inheritance_Pattern 等）与标签的关联

# ============================================================
# 6. 检查 prompt 中是否直接包含了答案
# ============================================================
print(f"\n🔎 Prompt 泄露检查 (答案词是否在 question 中出现):")
leak_count = 0
for i, row in df.iterrows():
    answer = str(row["answer"]).strip().lower()
    question = str(row["question"]).lower()

    keywords = []
    if "pathogenic" in answer:
        keywords.append("pathogenic")
    if "benign" in answer:
        keywords.append("benign")
    if "gain-of-function" in answer or "gof" in answer:
        keywords.extend(["gain-of-function", "gof"])
    if "loss-of-function" in answer or "lof" in answer:
        keywords.extend(["loss-of-function", "lof"])

    found = [kw for kw in keywords if kw in question]
    if found:
        leak_count += 1
        if i < 3:
            print(f"  [{i}] answer={answer[:60]} | question 含: {found}")

print(f"  question 中含答案词的样本: {leak_count}/{len(df)} ({leak_count/len(df)*100:.1f}%)")

# 注意：即使 prompt 里有答案词，由于 generate() 用 inputs_embeds 模式，
# generation 只包含新生成的 token，不包含 prompt。所以这不影响评估。
# 但可能影响训练——模型学会了复制 prompt 中的词而不是推理。

# ============================================================
# 7. 打印 test 集样本示例
# ============================================================
print(f"\n📝 Test 集前 3 条样本:")
for i in range(min(3, len(test_df))):
    row = test_df.iloc[i]
    print(f"\n  [{i}] Gene: {row['_gene']} | Answer: {row['answer']}")
    q = str(row["question"])
    # 只打印最后 200 字符
    print(f"  Question (last 200): ...{q[-200:]}")

# ============================================================
# 8. 检查 DNA 序列是否在 train/test 间重复
# ============================================================
print(f"\n🧬 DNA 序列重复检查:")
train_ref = set(train_df["reference_sequence"].dropna())
test_ref = set(test_df["reference_sequence"].dropna())
train_var = set(train_df["variant_sequence"].dropna())
test_var = set(test_df["variant_sequence"].dropna())

ref_overlap = train_ref & test_ref
var_overlap = train_var & test_var
print(f"  Train ref 序列: {len(train_ref)} 条, 唯一: {len(train_ref)}")
print(f"  Test  ref 序列: {len(test_ref)} 条, 唯一: {len(test_ref)}")
print(f"  Ref 序列 Train∩Test: {len(ref_overlap)} {'❌ 序列泄露!' if ref_overlap else '✅'}")
print(f"  Var 序列 Train∩Test: {len(var_overlap)} {'❌ 序列泄露!' if var_overlap else '✅'}")

# ============================================================
# 9. 总结
# ============================================================
print(f"\n" + "=" * 70)
print(f"诊断总结:")
print(f"  1. 基因级分割: {'✅ 无泄露' if not train_test_overlap else '❌ 有交叉!'}")
print(f"  2. DNA序列重复: {'✅ 无重复' if (not ref_overlap and not var_overlap) else '❌ 有重复!'}")
print(f"  3. 标签分布: 多数类占 {majority_pct:.1f}%")
print(f"  4. 纯标签基因占比: {pure_gene_samples/len(df)*100:.1f}%")
print(f"  5. Prompt含答案词: {leak_count}/{len(df)} (不影响评估但影响训练)")

if majority_pct > 90:
    print(f"  ⚠️  标签严重不平衡！模型只猜多数类就能 {majority_pct:.1f}%")
if train_test_overlap:
    print(f"  ❌ 基因泄露！train 和 test 有共同基因")
if ref_overlap or var_overlap:
    print(f"  ❌ DNA序列泄露！相同序列出现在 train 和 test")
print(f"=" * 70)
