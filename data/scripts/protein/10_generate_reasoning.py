#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 10 步：使用 API 生成严格 JSON 推理链，并转换为 SFT 训练目标。

核心设计：
  1. 两个阶段分别使用不同的 6 步推理框架。
  2. API 输出严格 JSON，字段为 reasoning_steps、final_synthesis、final_answer。
  3. 每一步固定包含 evidence、interpretation、mechanism_implication。
  4. SFT 使用扁平化文本：Step n: evidence interpretation mechanism_implication。
  5. 默认读取 09b 抽样后的文件，避免对全量数据调用 API。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from openai import OpenAI


ALLOWED_PREFIXES = (
    "[Supports LOF]",
    "[Supports GOF]",
    "[Supports Pathogenic]",
    "[Supports Neutral]",
    "[Ambiguous]",
    "[Context Dependent]",
)
NO_EVIDENCE_PLACEHOLDER = "[No usable evidence provided for this evidence group]"
SCHEMA_LEAK_PHRASES = (
    "no new features",
    "synthesize steps 1-5 only",
)
UNSUPPORTED_SPECIFIC_CLAIM_TERMS = (
    "known pathogenic",
    "reported variant",
    "reported variants",
    "literature",
    "patient",
    "patients",
    "cohort",
    "hotspot",
)
LABEL_CONDITIONING_LEAK_TERMS = (
    "provided correct classification",
    "correct classification",
    "provided label",
    "target label",
    "gold label",
    "authoritative target",
    "curated label",
)
CONFLICT_LANGUAGE_TERMS = (
    "mixed",
    "conflicting",
    "although",
    "while",
    "whereas",
    "however",
    "despite",
    "opposing",
    "opposite",
    "counter",
    "lof-leaning",
    "gof-leaning",
    "supports lof",
    "supports gof",
)
DECISIVE_LANGUAGE_TERMS = (
    "decisive evidence layer",
    "decisive layer",
    "decisive",
    "overriding",
    "overrides",
    "outweigh",
    "outweighs",
    "stronger evidence",
    "stronger layer",
    "priority",
    "final mechanism follows",
    "final_answer follows",
)

STAGE_SCHEMAS = {
    1: {
        "task": "Pathogenic vs Neutral",
        "valid_answers": ("Pathogenic", "Benign", "Neutral"),
        "steps": [
            {
                "step": 1,
                "category": "Variant consequence",
                "goal": "What is the immediate DNA, RNA, or protein consequence of this variant?",
                "features": ["Consequence", "REF", "ALT", "Amino_acids", "Protein_position", "in_last_exon"],
            },
            {
                "step": 2,
                "category": "Population tolerance",
                "goal": "Is this variant tolerated in human population data?",
                "features": ["AF", "MAX_AF"],
            },
            {
                "step": 3,
                "category": "Evolutionary constraint",
                "goal": "Is the affected site evolutionarily constrained across species?",
                "features": ["GERP++_RS", "phyloP100way_vertebrate"],
                "forbidden_features": ["CADD_phred"],
            },
            {
                "step": 4,
                "category": "Molecular damage prediction",
                "goal": "Do computational predictors support molecular damage?",
                "features": ["AlphaMissense_score", "MutPred_score", "CADD_phred"],
            },
            {
                "step": 5,
                "category": "Functional or structural context",
                "goal": "Is the variant located in a functional, structural, or gene-level sensitive context?",
                "features": [
                    "DOMAINS",
                    "Functional_Site",
                    "Secondary_Structure",
                    "AlphaFold_RSA",
                    "ESM_DDG_Score",
                    "Spatial_Density_10A",
                    "pLI",
                    "Haploinsufficiency_Score",
                ],
            },
            {
                "step": 6,
                "category": "Pathogenicity synthesis",
                "goal": "Synthesize the previous evidence to support Pathogenic or Neutral/Benign classification.",
                "features": ["Synthesis of steps 1-5"],
            },
        ],
    },
    2: {
        "task": "GOF vs LOF",
        "valid_answers": ("Gain-of-Function (GOF)", "Loss-of-Function (LOF)"),
        "steps": [
            {
                "step": 1,
                "category": "Initial molecular consequence",
                "goal": "What direct molecular consequence does this DNA or protein change first create?",
                "features": ["Consequence", "Amino_acids", "Protein_position"],
            },
            {
                "step": 2,
                "category": "Transcript and translation consequence",
                "goal": "Does this variant suggest NMD, truncation, frameshift, splice disruption, or preserved full-length protein production?",
                "features": [
                    "Consequence",
                    "in_last_exon",
                    "NMD_predicted",
                    "protein_truncation_percent",
                ],
                "forbidden_features": ["ESM_DDG_Score"],
            },
            {
                "step": 3,
                "category": "Functional region context",
                "goal": "If the protein product remains interpretable, which functional region does this variant affect?",
                "features": [
                    "DOMAINS",
                    "Functional_Site",
                    "Protein_position",
                    "Secondary_Structure",
                    "AlphaFold_RSA",
                    "Spatial_Density_10A",
                ],
            },
            {
                "step": 4,
                "category": "Post-translational biophysical effect",
                "goal": "What is the post-translational biophysical effect on the folded protein?",
                "features": ["ESM_DDG_Score", "AlphaFold_RSA", "Isoelectric_diff", "Molecular_weight"],
                "exclusive_features": ["ESM_DDG_Score"],
            },
            {
                "step": 5,
                "category": "Gene function and pathway context",
                "goal": "Given the gene's normal function, does the evidence suggest reduced function, abnormal activation, or dysregulated function?",
                "features": [
                    "Gene_Normal_Function",
                    "Pathway_Context",
                    "pLI",
                    "Haploinsufficiency_Score",
                    "Inheritance_Pattern",
                ],
            },
            {
                "step": 6,
                "category": "Mechanism synthesis",
                "goal": "Synthesize the previous five steps to support GOF or LOF.",
                "features": ["Synthesis of steps 1-5"],
            },
        ],
    },
}


parser = argparse.ArgumentParser(description="第 10 步：生成 JSON 推理链并扁平化为 SFT 文本")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="1=Pathogenic vs Neutral，2=GOF vs LOF")
parser.add_argument("--api_key", type=str,
                    default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                    help="API Key，也可以使用 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")
parser.add_argument("--base_url", type=str, default="https://linxi.chat/v1",
                    help="兼容 OpenAI Chat Completions 的 API 地址")
parser.add_argument("--model", type=str, default="deepseek-chat", help="模型名称")
parser.add_argument("--start", type=int, default=0, help="起始行号，含该行")
parser.add_argument("--end", type=int, default=-1, help="结束行号，不含该行；-1 表示到文件末尾")
parser.add_argument("--workers", type=int, default=5, help="并发线程数")
parser.add_argument("--max_retries", type=int, default=3, help="单条样本最大重试次数")
parser.add_argument("--temperature", type=float, default=0.2, help="生成温度，推理链建议保持较低")
parser.add_argument("--max_tokens", type=int, default=1800, help="单条样本最大输出 token")
args = parser.parse_args()

if not args.api_key:
    raise ValueError("请通过 --api_key 或 DEEPSEEK_API_KEY/OPENAI_API_KEY 环境变量提供 API Key")

BASE_DIR = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed"
if args.stage == 1:
    INPUT_CSV = os.path.join(BASE_DIR, "09b_BioReason_protein_Stage1_Binary_sampled.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "10_BioReason_protein_Stage1_Binary_Reasoning.csv")
else:
    INPUT_CSV = os.path.join(BASE_DIR, "09b_BioReason_protein_Stage2_GOF_LOF_sampled.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "10_BioReason_protein_Stage2_GOF_LOF_Reasoning.csv")

CHECKPOINT_CSV = OUTPUT_CSV.replace(".csv", "_checkpoint.csv")
ACTIVE_SCHEMA = STAGE_SCHEMAS[args.stage]


def build_system_prompt() -> str:
    schema_steps = json.dumps(ACTIVE_SCHEMA["steps"], ensure_ascii=False, indent=2)
    valid_answers = ", ".join(ACTIVE_SCHEMA["valid_answers"])

    stage_specific_rules = ""
    if args.stage == 1:
        stage_specific_rules = """
Stage 1 rules:
1. The task is pathogenicity classification: Pathogenic vs Neutral/Benign.
2. Do not discuss GOF or LOF mechanism unless it is explicitly needed to explain pathogenicity; the final answer must stay at pathogenicity level.
3. Step 6 must synthesize only steps 1-5 and support Pathogenic or Neutral/Benign.
4. Prefer mechanism_implication prefixes [Supports Pathogenic], [Supports Neutral], [Ambiguous], or [Context Dependent].
"""
    else:
        stage_specific_rules = """
Stage 2 rules:
1. The task is mechanism classification: Gain-of-Function (GOF) vs Loss-of-Function (LOF).
2. Step 2 is only about transcript/translation/product integrity. Do not use ESM_DDG_Score in Step 2.
3. Step 4 is only about post-translational folded-protein biophysics. ESM_DDG_Score belongs here only.
4. Step 5 should use Gene_Normal_Function and gene-level context. Use Pathway_Context only if it is explicitly present in the variant features.
5. Step 6 must synthesize only steps 1-5 and support the provided correct final_answer.
6. Prefer mechanism_implication prefixes [Supports LOF], [Supports GOF], [Ambiguous], or [Context Dependent].
7. If evidence supports both GOF and LOF, Step 6 must explicitly state the mixed/conflicting evidence, identify which steps support each direction, name the decisive evidence layer, and explain why final_answer follows that layer.
8. Decisive evidence layer priority for Stage 2 is:
   protein product loss/NMD/truncation > preserved protein with functional-region effect > post-translational biophysical damage > gene-function/pathway context > gene-level constraint/inheritance.
9. Gene-level constraint and inheritance pattern must not be the sole decisive layer for GOF or LOF.
10. For a GOF final_answer with a preserved missense/inframe protein, do not let a destabilizing ESM_DDG_Score alone force LOF. Treat it as conflicting LOF-leaning evidence, then explain how preserved protein product plus functional-region/gene-function context can support an altered activity, gating, regulation, or interaction mechanism.
11. For a LOF final_answer with truncation, likely NMD, frameshift, or severe protein loss, protein product integrity is usually the decisive layer even if gene context is mixed.
"""

    return f"""
You are an expert computational biologist building supervised training data for variant reasoning.

Return exactly one valid JSON object. Do not use markdown fences. Do not add extra text.

Required JSON schema:
{{
  "reasoning_steps": [
    {{
      "step": 1,
      "category": "...",
      "evidence": "...",
      "interpretation": "...",
      "mechanism_implication": "[Allowed Prefix] ..."
    }}
  ],
  "final_synthesis": "...",
  "final_answer": "..."
}}

You must output exactly 6 reasoning_steps, using this stage-specific plan:
{schema_steps}

This is label-conditioned rationale generation, not open-ended prediction. The correct classification in the user message is the authoritative target label.
The final_answer must exactly equal one of the provided valid labels and must exactly match the correct classification in the user message.
Never change final_answer away from the provided correct classification. If some features appear to support the opposite label, explain the conflict and build a biologically plausible rationale for the provided correct classification.
Valid labels for this stage: {valid_answers}

General rules:
1. Each step must move from evidence to interpretation to mechanism_implication.
2. evidence should cite concrete feature labels and values exactly as shown in the prompt, for example "Consequence: missense variant; Amino acid change: R to H".
3. Use only evidence explicitly present in the variant features. Do not invent disease names, pathways, interactions, expression, experiments, or missing feature values.
4. If a feature is absent, do not write that it is absent unless the prompt explicitly says it is absent.
5. Do not add a question field. The JSON object must not contain "question".
6. Every mechanism_implication must start with exactly one of:
   [Supports LOF]
   [Supports GOF]
   [Supports Pathogenic]
   [Supports Neutral]
   [Ambiguous]
   [Context Dependent]
7. The final_synthesis should be one concise sentence integrating the six-step chain.
8. If an Evidence group states "[No usable evidence provided for this evidence group]", you MUST explicitly cite this exact bracketed text in the evidence field for that step. DO NOT hallucinate or infer missing features. ABSOLUTELY DO NOT treat the absence of evidence as proof of a Benign, Pathogenic, GOF, or LOF effect. The interpretation must state that data is insufficient for that step, and the mechanism_implication MUST start with [Ambiguous].
9. Step 6 evidence must be written as "Synthesis of steps 1-5". Do not copy schema instructions such as "no new features; synthesize steps 1-5 only".
10. Use general biological principles to interpret the provided features, but do not introduce specific literature, cohort, disease, hotspot, patient, reported-variant, or variant-history claims unless they are explicitly present in the prompt.
11. Do not mention that the label was provided, curated, authoritative, or known. The output should read like a normal biological rationale, not like an explanation of the data-construction process.
{stage_specific_rules}
"""


SYSTEM_PROMPT = build_system_prompt()


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def placeholder_steps_from_question(question: str) -> set[int]:
    """识别 prompt 中哪些 Evidence group 使用了空证据占位符。"""
    steps: set[int] = set()
    current_step: int | None = None
    current_block: list[str] = []

    def flush_block() -> None:
        if current_step is not None and NO_EVIDENCE_PLACEHOLDER in "\n".join(current_block):
            steps.add(current_step)

    for line in str(question).splitlines():
        match = re.match(r"^## Evidence group\s+(\d+):", line)
        if match:
            flush_block()
            current_step = int(match.group(1))
            current_block = [line]
        elif current_step is not None:
            current_block.append(line)
    flush_block()
    return steps


def contains_unsupported_specific_claim(text: str, question: str) -> str:
    """返回未被 prompt 支持的具体历史/文献断言关键词；空字符串表示未命中。"""
    text_lower = text.lower()
    question_lower = question.lower()
    for term in UNSUPPORTED_SPECIFIC_CLAIM_TERMS:
        if term in text_lower and term not in question_lower:
            return term
    return ""


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象；不接受无法解析的文本。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_reasoning_json(payload: dict[str, Any], expected_answer: str, question: str = "") -> list[str]:
    """返回错误列表；空列表表示通过。"""
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ["输出不是 JSON 对象"]

    forbidden_top_level = {"steps", "answer", "question"}
    extra_forbidden = sorted(forbidden_top_level & set(payload))
    if extra_forbidden:
        errors.append(f"包含旧版或不允许的顶层字段: {extra_forbidden}")

    if payload.get("final_answer") != expected_answer:
        errors.append("final_answer 与标签不一致")

    if not normalize_text(payload.get("final_synthesis")):
        errors.append("缺少 final_synthesis")

    steps = payload.get("reasoning_steps")
    if not isinstance(steps, list) or len(steps) != 6:
        errors.append("reasoning_steps 必须是长度为 6 的列表")
        return errors

    expected_steps = ACTIVE_SCHEMA["steps"]
    placeholder_steps = placeholder_steps_from_question(question)
    prompt_question = question or ""
    step_direction_prefixes: dict[int, str] = {}
    for expected, step_obj in zip(expected_steps, steps):
        if not isinstance(step_obj, dict):
            errors.append(f"step {expected['step']} 不是对象")
            continue

        if "question" in step_obj:
            errors.append(f"step {expected['step']} 不应包含 question 字段")

        if step_obj.get("step") != expected["step"]:
            errors.append(f"step 编号错误，应为 {expected['step']}")

        if step_obj.get("category") != expected["category"]:
            errors.append(f"step {expected['step']} category 错误")

        for field in ("evidence", "interpretation", "mechanism_implication"):
            if not normalize_text(step_obj.get(field)):
                errors.append(f"step {expected['step']} 缺少 {field}")

        implication = normalize_text(step_obj.get("mechanism_implication"))
        if not implication.startswith(ALLOWED_PREFIXES):
            errors.append(f"step {expected['step']} mechanism_implication 前缀不合法")
        else:
            step_direction_prefixes[expected["step"]] = next(prefix for prefix in ALLOWED_PREFIXES if implication.startswith(prefix))

        joined_step_text = " ".join(
            normalize_text(step_obj.get(k))
            for k in ("evidence", "interpretation", "mechanism_implication")
        )
        unsupported_term = contains_unsupported_specific_claim(joined_step_text, prompt_question)
        if unsupported_term:
            errors.append(f"step {expected['step']} 包含 prompt 未提供的具体历史/文献断言: {unsupported_term}")
        lower_joined_step_text = joined_step_text.lower()
        leaked_label_term = next((term for term in LABEL_CONDITIONING_LEAK_TERMS if term in lower_joined_step_text), "")
        if leaked_label_term:
            errors.append(f"step {expected['step']} 泄露了标签条件构建痕迹: {leaked_label_term}")

        if expected["step"] == 6:
            lower_step6 = lower_joined_step_text
            if any(phrase in lower_step6 for phrase in SCHEMA_LEAK_PHRASES):
                errors.append("step 6 不应复制 schema 规则文本")
            evidence = normalize_text(step_obj.get("evidence"))
            if "Synthesis of steps 1-5" not in evidence:
                errors.append('step 6 evidence 必须写为或包含 "Synthesis of steps 1-5"')
            if args.stage == 1:
                if expected_answer == "Pathogenic" and not implication.startswith("[Supports Pathogenic]"):
                    errors.append("Stage 1 step 6 必须支持 Pathogenic")
                if expected_answer in {"Benign", "Neutral"} and not implication.startswith("[Supports Neutral]"):
                    errors.append("Stage 1 step 6 必须支持 Neutral/Benign")
            if args.stage == 2:
                if expected_answer == "Gain-of-Function (GOF)" and not implication.startswith("[Supports GOF]"):
                    errors.append("Stage 2 step 6 必须支持 GOF")
                if expected_answer == "Loss-of-Function (LOF)" and not implication.startswith("[Supports LOF]"):
                    errors.append("Stage 2 step 6 必须支持 LOF")

        if expected["step"] in placeholder_steps:
            evidence = normalize_text(step_obj.get("evidence"))
            if NO_EVIDENCE_PLACEHOLDER not in evidence:
                errors.append(f"step {expected['step']} 必须在 evidence 中引用空证据占位符")
            if not implication.startswith("[Ambiguous]"):
                errors.append(f"step {expected['step']} 为空证据组时 mechanism_implication 必须为 [Ambiguous]")

        if args.stage == 1 and expected["step"] == 3:
            joined = " ".join(normalize_text(step_obj.get(k)) for k in ("evidence", "interpretation", "mechanism_implication"))
            if "CADD_phred" in joined or "CADD" in joined:
                errors.append("Stage 1 Step 3 不允许使用 CADD_phred；CADD 只属于 Step 4")

        if args.stage == 2 and expected["step"] == 2:
            joined = " ".join(normalize_text(step_obj.get(k)) for k in ("evidence", "interpretation", "mechanism_implication"))
            if "ESM_DDG_Score" in joined or "ESM ddG" in joined:
                errors.append("Stage 2 Step 2 不允许使用 ESM_DDG_Score")

    if args.stage == 2 and isinstance(steps, list) and len(steps) == 6:
        first_five = [step_direction_prefixes.get(i, "") for i in range(1, 6)]
        has_gof = "[Supports GOF]" in first_five
        has_lof = "[Supports LOF]" in first_five
        if has_gof and has_lof:
            step6 = steps[5]
            step6_text = " ".join(
                normalize_text(step6.get(k))
                for k in ("evidence", "interpretation", "mechanism_implication")
            ).lower()
            if not any(term in step6_text for term in CONFLICT_LANGUAGE_TERMS):
                errors.append("Stage 2 证据冲突时 step 6 必须明确说明冲突证据")
            if not any(term in step6_text for term in DECISIVE_LANGUAGE_TERMS):
                errors.append("Stage 2 证据冲突时 step 6 必须说明决定性证据层或等价裁决理由")

    return errors


def flatten_reasoning(payload: dict[str, Any]) -> str:
    """按最终 SFT 规则扁平化。"""
    lines: list[str] = []
    for step_obj in payload["reasoning_steps"]:
        step_no = step_obj["step"]
        evidence = normalize_text(step_obj.get("evidence"))
        interpretation = normalize_text(step_obj.get("interpretation"))
        implication = normalize_text(step_obj.get("mechanism_implication"))
        lines.append(f"Step {step_no}: {evidence} {interpretation} {implication}")
    lines.append("")
    lines.append(f"Answer: {payload['final_answer']}")
    return "\n".join(lines)


def build_user_prompt(question: str, answer: str) -> str:
    return f"""
Task: {ACTIVE_SCHEMA["task"]}

Variant features:
{question}

Correct classification:
{answer}

Generate the six-step JSON reasoning chain according to the stage-specific schema.
"""


def call_model(client: OpenAI, question: str, answer: str) -> dict[str, Any] | None:
    user_prompt = build_user_prompt(question, answer)
    last_error = ""

    for attempt in range(1, args.max_retries + 1):
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
            text = response.choices[0].message.content or ""
            payload = extract_json_object(text)
            errors = validate_reasoning_json(payload, answer, question)
            if not errors:
                return payload
            last_error = "; ".join(errors)
        except Exception as exc:
            last_error = str(exc)

        print(f"  尝试 {attempt}/{args.max_retries} 未通过: {last_error}", flush=True)
        if attempt < args.max_retries:
            time.sleep(2 * attempt)

    return None


def process_one(client: OpenAI, row: pd.Series, idx: int) -> tuple[int, str, str, str, int]:
    question = normalize_text(row["question"])
    answer = normalize_text(row["answer"])
    payload = call_model(client, question, answer)

    if payload is None:
        return idx, "", "", "failed", 0

    reasoning_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    reasoning_sft = flatten_reasoning(payload)
    return idx, reasoning_json, reasoning_sft, "ok", len(reasoning_sft)


def load_existing_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("reasoning_json", "reasoning_sft", "reasoning_status"):
        if col not in df.columns:
            df[col] = pd.NA

    if not os.path.exists(CHECKPOINT_CSV):
        return df

    cp = pd.read_csv(CHECKPOINT_CSV)
    for col in ("reasoning_json", "reasoning_sft", "reasoning_status"):
        if col in cp.columns:
            df[col] = cp[col]
    print(f"已加载断点文件: {CHECKPOINT_CSV}", flush=True)
    return df


def main() -> None:
    print("=" * 80)
    print(f"第 10 步：生成严格 JSON 推理链并扁平化，stage={args.stage}")
    print(f"输入: {INPUT_CSV}")
    print(f"输出: {OUTPUT_CSV}")
    print(f"模型: {args.model} | 并发: {args.workers}")
    print("=" * 80)

    df = pd.read_csv(INPUT_CSV)
    df = load_existing_checkpoint(df)

    end_idx = args.end if args.end > 0 else len(df)
    end_idx = min(end_idx, len(df))
    indices = [
        i for i in range(args.start, end_idx)
        if pd.isna(df.at[i, "reasoning_sft"]) or not normalize_text(df.at[i, "reasoning_sft"])
    ]

    if not indices:
        print("指定范围内没有待生成样本，直接写出最终文件。", flush=True)
        df.to_csv(OUTPUT_CSV, index=False)
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    lock = threading.Lock()
    completed = 0
    ok_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, client, df.iloc[i], i): i for i in indices}

        for future in as_completed(futures):
            idx, reasoning_json, reasoning_sft, status, chars = future.result()
            with lock:
                completed += 1
                if status == "ok":
                    ok_count += 1
                    df.at[idx, "reasoning_json"] = reasoning_json
                    df.at[idx, "reasoning_sft"] = reasoning_sft
                    df.at[idx, "reasoning_status"] = "ok"
                else:
                    failed_count += 1
                    df.at[idx, "reasoning_status"] = "failed"

                print(f"[{completed}/{len(indices)}] idx={idx} status={status} chars={chars}", flush=True)

                if completed % 50 == 0 or completed == len(indices):
                    df.to_csv(CHECKPOINT_CSV, index=False)
                    print(f"已保存断点: {CHECKPOINT_CSV}", flush=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print("=" * 80)
    print("第 10 步完成")
    print(f"成功: {ok_count} | 失败: {failed_count} | 输出: {OUTPUT_CSV}")
    print("失败样本不会被伪造 fallback；可保留 failed 状态后续重跑。")
    print("=" * 80)


if __name__ == "__main__":
    main()
