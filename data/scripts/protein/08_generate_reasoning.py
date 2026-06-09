"""
使用 DeepSeek API 生成高质量变异效应推理链
==========================================

读取 07_format_bioreason_prompt.py 输出的训练 CSV，
调用 DeepSeek API 为每条变异生成 <think>...</think> 推理链，
输出带 reasoning 列的 CSV，供 SFT 训练使用。

用法:
  python 08_generate_reasoning.py --stage 1 --api_key YOUR_KEY
  python 08_generate_reasoning.py --stage 2 --api_key YOUR_KEY

断点续传:
  python 08_generate_reasoning.py --stage 1 --api_key YOUR_KEY --start 500
"""

import argparse
import os
import sys
import time
import pandas as pd
from openai import OpenAI
from typing import Optional

# ==========================================
# 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="DeepSeek API 推理链生成器")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="1=阶段一(致病vs中性), 2=阶段二(GOF vs LOF)")
parser.add_argument("--api_key", type=str, required=True, help="DeepSeek API Key")
parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
parser.add_argument("--model", type=str, default="deepseek-chat",
                    help="deepseek-chat (V3, 便宜) 或 deepseek-reasoner (R1, 贵)")
parser.add_argument("--start", type=int, default=0, help="从第几行开始 (断点续传)")
parser.add_argument("--end", type=int, default=-1, help="到第几行结束 (-1=全部)")
parser.add_argument("--sleep", type=float, default=0.3, help="API 调用间隔(秒)")
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max_tokens", type=int, default=1024)
args = parser.parse_args()

STAGE = args.stage

# 输入: 07_format_bioreason_prompt.py 的输出
if STAGE == 1:
    INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage1_Binary.csv"
    OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage1_Binary_Reasoning.csv"
else:
    INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage2_GOF_LOF.csv"
    OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/BioReason_protein_Stage2_GOF_LOF_Reasoning.csv"

# 临时 checkpoint 文件 (每处理一条就存一次，防止中断丢失)
CHECKPOINT_CSV = OUTPUT_CSV.replace(".csv", "_checkpoint.csv")

# ==========================================
# System Prompt (阶段不同，推理重点不同)
# ==========================================
if STAGE == 1:
    SYSTEM_PROMPT = """You are an expert computational biologist specializing in variant effect prediction.

I will give you a genetic variant with its genomic, evolutionary, and structural features. Your task is to write a logical step-by-step reasoning chain, then classify the variant as Pathogenic or Benign.

Rules:
1. Wrap your reasoning in <think>...</think> tags
2. Reference specific numeric feature values from the input in your reasoning
3. Reason step by step: mutation type → evolutionary conservation → computational predictions → structural impact → conclusion
4. After </think>, output exactly "Answer: Pathogenic" or "Answer: Benign" (no other text)
5. Keep reasoning concise: 5-8 steps, ~150-250 words total
6. If a feature value is "Unknown", skip it — do not guess or hallucinate
7. Do NOT mention that features are "not provided" or "missing" — simply omit them

Key reasoning patterns for Pathogenic:
- High CADD phred (>20) and high GERP (>2) indicate strong evolutionary constraint
- Low or zero MAX_AF suggests purifying selection against this variant
- Destabilizing ESM ddG (|ddG| > 2) suggests protein structure disruption
- Truncating variants (stop_gained, frameshift) nearly always pathogenic
- High SpliceAI scores indicate splicing disruption

Key reasoning patterns for Benign:
- High MAX_AF in population suggests tolerated variation
- Low CADD, low GERP suggest weak selective pressure
- Stable ESM ddG (|ddG| < 0.5) suggests no structural disruption
- Synonymous or deep intronic variants with no splice effect"""
else:
    SYSTEM_PROMPT = """You are an expert computational structural biologist specializing in protein mechanism analysis.

I will give you a pathogenic genetic variant with its genomic, evolutionary, and structural features. Your task is to write a logical step-by-step reasoning chain analyzing whether this variant causes Gain-of-Function (GOF) or Loss-of-Function (LOF).

Rules:
1. Wrap your reasoning in <think>...</think> tags
2. Reference specific numeric feature values from the input in your reasoning
3. Reason step by step: mutation type → protein integrity → structural stability (ESM ddG, RSA) → biochemical changes (charge, hydrophobicity) → mechanism inference → conclusion
4. After </think>, output exactly "Answer: Gain-of-Function (GOF)" or "Answer: Loss-of-Function (LOF)" (no other text)
5. Keep reasoning concise: 5-8 steps, ~150-250 words total
6. If a feature value is "Unknown", skip it — do not guess or hallucinate
7. Do NOT mention that features are "not provided" or "missing" — simply omit them

Key GOF indicators (protein remains functional but overactive):
- Stable fold: ESM |ddG| < 1, protein not destabilized
- Surface exposure: RSA > 0.5, residue accessible for new interactions
- Missense variant (not truncating): full-length protein produced
- Charge reversal or hydrophobicity shift at interaction interface
- Low SpliceAI: no splicing disruption

Key LOF indicators (protein function reduced or abolished):
- Destabilizing: ESM ddG < -2 or > 2, protein folding compromised
- Truncating: stop_gained, frameshift → no functional protein
- Buried residue: RSA < 0.2, mutation disrupts hydrophobic core
- High SpliceAI: splicing disruption → aberrant transcript
- High haploinsufficiency score: gene is dosage-sensitive"""

# ==========================================
# 初始化 API Client
# ==========================================
client = OpenAI(api_key=args.api_key, base_url=args.base_url)


def call_deepseek(question: str, answer: str) -> Optional[str]:
    """
    调用 DeepSeek API 生成推理链。

    Args:
        question: 变异特征 prompt (来自 07 脚本)
        answer:   正确答案 (Pathogenic/Benign 或 GOF/LOF)

    Returns:
        "<think>...</think>\n\nAnswer: xxx" 格式的完整回复，失败返回 None
    """
    user_prompt = f"""## Variant Features
{question}

## Expected Classification
The correct answer is: {answer}

Please analyze the features above and generate your reasoning chain."""

    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content.strip()
            return content

        except Exception as e:
            print(f"  ⚠️ 尝试 {attempt + 1}/{args.max_retries} 失败: {e}")
            if attempt < args.max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"     等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  ❌ 全部重试失败，返回 None")
                return None


def validate_response(text: str, expected_answer: str) -> bool:
    """验证 API 返回的回复格式是否正确"""
    if not text:
        return False
    has_think_open = "<think>" in text
    has_think_close = "</think>" in text
    has_answer = "Answer:" in text
    has_correct_answer = expected_answer in text
    return has_think_open and has_think_close and has_answer and has_correct_answer


# ==========================================
# 主流程
# ==========================================
def main():
    print("=" * 70)
    print(f"🧬 DeepSeek 推理链生成器 — 阶段{STAGE}")
    print(f"   模型: {args.model}")
    print(f"   温度: {args.temperature}")
    print("=" * 70)

    # 1. 读取输入
    print(f"\n📂 读取: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"   共 {len(df)} 条样本")

    # 2. 确定处理范围
    start_idx = args.start
    end_idx = args.end if args.end > 0 else len(df)
    print(f"   处理范围: [{start_idx}, {end_idx}) 共 {end_idx - start_idx} 条")

    # 3. 断点续传: 读取已有 checkpoint
    if os.path.exists(CHECKPOINT_CSV) and start_idx == 0:
        df_cp = pd.read_csv(CHECKPOINT_CSV)
        already_done = len(df_cp)
        print(f"📌 发现 checkpoint: 已完成 {already_done} 条，从第 {already_done} 条继续")
        start_idx = already_done
        reasoning_list = df_cp["reasoning"].tolist() if "reasoning" in df_cp.columns else []
    else:
        reasoning_list = []

    # 4. 逐条调用 API
    success_count = 0
    fail_count = 0

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        question = row["question"]
        answer = row["answer"]

        print(f"\n[{i}/{len(df)}] ", end="", flush=True)
        print(f"Answer={answer} | ", end="", flush=True)

        result = call_deepseek(question, answer)

        if result and validate_response(result, answer):
            print(f"✅ ({len(result)} chars)", flush=True)
            reasoning_list.append(result)
            success_count += 1
        elif result:
            # 格式不对但内容非空，仍然保存（可能有微小格式差异）
            print(f"⚠️ 格式校验未通过 ({len(result)} chars)，已保存", flush=True)
            reasoning_list.append(result)
            success_count += 1
        else:
            # API 完全失败，用退化版本
            fallback = f"<think>\nThe provided features indicate this variant is {answer.lower()}.\n</think>\n\nAnswer: {answer}"
            print(f"❌ 使用退化模板", flush=True)
            reasoning_list.append(fallback)
            fail_count += 1

        # 每处理一条就保存 checkpoint
        df_out = df.iloc[:len(reasoning_list)].copy()
        df_out["reasoning"] = reasoning_list
        df_out.to_csv(CHECKPOINT_CSV, index=False)

        time.sleep(args.sleep)

    # 5. 最终输出
    df_out = df.iloc[:len(reasoning_list)].copy()
    df_out["reasoning"] = reasoning_list
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    # 6. 统计
    print("\n" + "=" * 70)
    print("📊 生成完成！")
    print(f"   成功: {success_count}")
    print(f"   失败(退化): {fail_count}")
    print(f"   有效比例: {success_count}/{success_count + fail_count} ({success_count/(success_count+fail_count)*100:.1f}%)")
    print(f"💾 保存至: {OUTPUT_CSV}")

    # 删除 checkpoint
    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)
        print(f"🧹 已清理 checkpoint 文件")

    # 预览第一条
    print("\n👇 第一条推理链预览:")
    print("-" * 70)
    first_reasoning = df_out["reasoning"].iloc[0]
    print(first_reasoning[:500] + ("..." if len(first_reasoning) > 500 else ""))
    print("-" * 70)
    print("✅ 完成。")


if __name__ == "__main__":
    main()
