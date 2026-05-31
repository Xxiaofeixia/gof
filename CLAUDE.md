# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

该项目是目的用利用BioReason中的框架区分中性突变，与疾病有关的lof和gof，采用bioreason里面的方法，利用LLM+DNA模型+特征注释

主要分为两个阶段 第一阶段是区分中性突变和致病性突变（gof/lof） 第二阶段是区分gof和lof

BioReason — 一个多模态 DNA-LLM，将 DNA 序列编码器（NucleotideTransformer 或 Evo2）与文本 LLM（Qwen3）结合，用于生物推理任务。DNA embedding 被投影到文本模型的 embedding 空间中，通过 `<|dna_pad|>` 占位符 token 注入。原项目连接：https://github.com/bowang-lab/BioReason

两个训练阶段：
1. **SFT**（监督微调）：PyTorch Lightning + LoRA，最大化正确答案的对数概率
2. **GRPO**（分组相对策略优化）：基于 TRL 的自定义训练器，配合 vLLM 共址部署，使用奖励函数对生成答案打分，执行 PPO 风格的裁剪更新

## 数据集构建

数据来源：
● Gofcards ：3162条GOF数据 去除个同一突变对应多个疾病的(1127)，最终有2032个突变gof数据
● GOF/LOF里面的(ClinVar-based)数据集：555条GOF，4530条LOF
（Vpatho和LoGoFunc采用的是GOF/LOF里面的(HGMD-based)数据集，数据更多，但是不好获取）
中性突变：1.从clinivar里面获取良性和可能良性 2.从 gnomAD 里面获取高频且不在clinvar里面的标记致病的 目前采用的第一种方式
处理流程：
首先合并gofcard中的gof和GOF/LOF(ClinVar-based)中记载gof基因，获得了618个基因的2394条数，然后将这些gof所在的618个基因为基础，提取GOF/LOF(ClinVar-based)在这些基因下的lof，并在gof和lof中删除在gofcard和GOF/LOF中记载冲突的基因15条 ，获得2379条gof 1088条lof
中性突变在ClinVar随机抽取与gof所在的618个基因下的被标记为良性或可能良性的3000条

最后获得基因
gof：2379个  lof：1088个 中性突变：3000个

加入特征：
基本信息：
SYMBOL (基因名)
Location (染色体物理坐标),
基因级别的特征：
●  Amino_acids (氨基酸改变) VEP 注释
●  Consequence (突变后果，如 missense),
● DOMAINS (UniProt 结合域) VEP 注释
● AlphaMissense_score VEP 注释
●  CADD_phred
●  GERP++_RS 
●  phyloP100way_vertebrate 
● MAX_AF (最高人群频率)
蛋白质级别特征：
● pLI (Probability of being LoF Intolerant)
● Inheritance_Pattern 
●  MOI (ClinGen 遗传模式) 
●  Haploinsufficiency_Score (gnomAD 单倍剂量不足概率)
● AlphaMissense_score
● Secondary_Structure  
● AlphaFold_RSA
● ESM_DDG_Score: ESM-2 大模型预测的热力学稳定性改变
● Spatial_Density_10A: 3D 空间 10 埃范围内的致病点聚类密度。

特征选择：
参考对生成的特征进行两阶段的p值判断，筛选出p<0.05的作为最终输入

将上述选择后的特征以提示词的形式融合到问题中去参与训练




DNA模型需要的特征：
● reference_sequence （突变前突变位点的4kb的DNA序列）
●  variant_sequence (突变后突变位点的4kb的DNA序列) 

采用的大模型和DNA模型：qwen3-4B和NT

## 仓库结构

```
llm_training/                  # 所有训练和模型代码
  bioreason/
    models/
      dna_llm.py               # DNALLMModel — 多模态模型：DNA 编码器 + 投影层 + 文本 LLM
      dna_only.py               # DNAClassifierModel — 纯 DNA 分类基线
      dna_vllm.py               # 基于 vLLM 的 DNALLMModel，仅用于推理（与训练分离）
      dl/processing_dl.py       # DLProcessor — 处理文本分词 + DNA 分词 + 占位符注入
      dl/chat_template_dl.py    # Qwen3 DNA-LLM 格式的 Jinja 对话模板
      evo2_tokenizer.py         # Evo2 分词器封装
    trainer/
      grpo_trainer.py           # DNALLMGRPOTrainer — GRPO 训练器（生成 → 打分 → 优势 → PPO 更新）
      grpo_config.py            # DNALLMGRPOConfig — 所有 GRPO 超参数
    dataset/
      kegg.py                   # KEGG 数据集格式化，qwen_dna_collate_fn（标签构建）
      variant_effect.py         # 变异效应数据集
      utils.py                  # DNA 截断工具
    dna_modules/
      dna_module.py             # DNABaseModule 抽象基类
      nucleotide_module.py      # NucleotideDNAModule — prompt 准备，奖励函数（xmlcount、format、correctness、concise）
    utils/                      # vLLM 同步、checkpoint 保存、DNA 工具
  train_dna_qwen.py             # 主 SFT 训练脚本（PyTorch Lightning）
  train_grpo.py                 # 主 GRPO 训练脚本（GRPOTrainer，加载 SFT checkpoint）
  train_dna_only.py             # 纯 DNA 分类基线
  train_llm_only.py             # 纯 LLM 消融实验（无 DNA 编码器）
  eval_kegg_dna_vllm.py         # 在 KEGG 测试集上评估训练好的 checkpoint
data/
  scripts/gene/                 # 基因级数据流水线（01_export → 03_build → 05_extract → 06_format）
  scripts/protein/              # 蛋白质级特征流水线（AlphaFold PDB、DSSP、ESM 评分、空间密度）
  scripts/gene/_archived/       # 旧流水线脚本（liftover、merge、audit）
```

## 训练说明
现在的项目存放在本地，真正的训练在服务器端，本地用于调试后在服务器端进行训练

## 核心架构概念

**DNA embedding 注入** — 核心多模态设计。DNA 序列由 DNA 分词器（NT 或 Evo2）分词，经过 DNA 编码器，通过 `dna_projection`（线性层）投影，然后注入文本 embedding 序列中所有 `<|dna_pad|>` token 出现的位置。这避免了将 DNA 转换为文本字符串，让 LLM 以连续 embedding 的形式处理 DNA。

**LoRA 训练策略** — DNA 编码器始终冻结；文本 LLM 在所有线性层上添加 LoRA 适配器；`dna_projection` 始终完全可训练（桥接层）。SFT checkpoint 在 GRPO 前将 LoRA 合并到基础权重中，然后 GRPO 添加新的 LoRA。

**vLLM 共址部署** — GRPO 期间，vLLM 和训练共享同一 GPU。训练器调用 `model.get_prompt_embeddings()`（注入 DNA embedding），然后将 `prompt_embeds` 传递给 vLLM 的 generate。vLLM 在优化器步骤期间休眠以释放内存，在生成和权重同步时唤醒。

**标签构建**（`qwen_dna_collate_fn`）— 只有 `<|im_start|>assistant\n` 和 `<|im_end|>` 标记之间的 token 参与 loss 计算；所有其他位置设为 -100（忽略）。

**GRPO 流程** — 每个生成批次为每个 prompt 生成 G 个响应；计算奖励（正确性、格式、xmlcount、简洁性）；优势 = 奖励 - 组内均值；PPO 裁剪 loss 更新策略。生成的补全结果在 `steps_per_generation` 个累积步骤中缓存。

## 常用命令

SFT 训练：
```bash
python llm_training/train_dna_qwen.py \
  --model_type dna-llm \
  --dataset_type kegg \
  --text_model_name Qwen/Qwen3-1.7B \
  --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
  --batch_size 1 --max_epochs 5 --num_gpus 1
```

GRPO 训练（需要 SFT checkpoint）：
```bash
python llm_training/train_grpo.py \
  --text_model_name Qwen/Qwen3-4B \
  --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
  --sft_checkpoint /path/to/sft/checkpoint \
  --output_dir /path/to/output
```

评估：
```bash
python llm_training/eval_kegg_dna_vllm.py \
  --ckpt_dir /path/to/checkpoint \
  --cache_dir /model-weights \
  --output_dir ./eval_results
```

纯 DNA 基线：
```bash
python llm_training/train_dna_only.py --dataset_type kegg --wandb_entity YOUR_ENTITY
```

下载模型/数据集：
```bash
python llm_training/download_models.py   # Qwen3 + NT 权重（使用 hf-mirror.com）
python llm_training/download_data.py     # KEGG 数据集下载到本地磁盘
```

## 依赖

核心包：`transformers`、`torch`、`pytorch-lightning`、`peft`、`trl`（定制 fork）、`vllm`、`datasets`、`wandb`、`evo2`、`vortex`（用于 Evo2 StripedHyena 主干网络）。项目假定运行环境为配备 CUDA GPU 的 Linux 集群，全程使用 `bf16-mixed` 精度。
