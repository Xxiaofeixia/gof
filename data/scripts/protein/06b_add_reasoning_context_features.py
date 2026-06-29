#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 06b 步：补充推理链构建所需的最小上下文特征。

新增列：
  Gene_Normal_Function
  Functional_Site
  NMD_predicted
  protein_truncation_percent

设计原则：
  1. 只补充 GOF/LOF 推理链真正需要的少量上下文，不堆叠同类预测分数。
  2. Gene_Normal_Function 和 Functional_Site 来自 UniProt reviewed 条目。
  3. NMD_predicted 只从 LOFTEE 注释解释，不再使用自写粗规则预测。
  4. protein_truncation_percent 从现有 VEP/蛋白位置注释派生。
  4. 原地更新 06_BIOREASON_with_Biochem.csv，避免下游脚本路径发散。
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INPUT_CSV = os.path.join(PROCESSED_DIR, "06_BIOREASON_with_Biochem.csv")
OUTPUT_CSV = INPUT_CSV
MAPPING_CSV = os.path.join(PROCESSED_DIR, "uniprot_mapping.csv")
CACHE_JSON = os.path.join(PROCESSED_DIR, "uniprot_reasoning_context_cache.json")
GENE_FUNCTION_MAP_CSV = os.path.join(PROCESSED_DIR, "gene_function_map.csv")

MAX_WORKERS = 8
UNIPROT_TIMEOUT = 20
REQUEST_HEADERS = {"User-Agent": "gof-bioreason-reasoning-context/1.0"}

SITE_PRIORITY = [
    "Active site",
    "Binding site",
    "Metal binding",
    "Site",
    "Modified residue",
    "Disulfide bond",
    "Cross-link",
    "Glycosylation",
    "Lipidation",
]
NO_SITE_PLACEHOLDER = "No annotated functional site at this position"

TRUNCATING_TERMS = {
    "stop_gained",
    "frameshift_variant",
}
LOFTEE_MISSING = {"", "-", "None", "none", "nan", "NaN"}


def load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_symbol(value: Any) -> str:
    return str(value).strip().upper()


def parse_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "-", "None", "nan", "Unknown"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def consequence_terms(value: Any) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def extract_position(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def trim_sentences(text: str, max_sentences: int = 2, max_chars: int = 450) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text:
        return "Function unknown"
    parts = re.split(r"(?<=[.;!?])\s+", text)
    short = " ".join(parts[:max_sentences]).strip()
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;") + "."
    return short or "Function unknown"


def load_uniprot_mapping() -> dict[str, str]:
    mapping_df = pd.read_csv(MAPPING_CSV, low_memory=False)
    mapping: dict[str, str] = {}
    for row in mapping_df.itertuples(index=False):
        symbol = normalize_symbol(row.SYMBOL)
        uid = str(row.UniProt_ID).strip()
        if symbol and uid and uid.lower() not in {"nan", "none", "-"}:
            mapping[symbol] = uid
    return mapping


def fetch_uniprot_context(uid: str, cache: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if uid in cache:
        return uid, cache[uid]

    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=UNIPROT_TIMEOUT, verify=False)
        if response.status_code != 200:
            payload = {"ok": False, "function": "Function unknown", "features": [], "length": None}
            cache[uid] = payload
            return uid, payload

        data = response.json()
        function_texts: list[str] = []
        for comment in data.get("comments", []):
            if comment.get("commentType") != "FUNCTION":
                continue
            for text_obj in comment.get("texts", []):
                value = text_obj.get("value")
                if value:
                    function_texts.append(value)

        sequence = data.get("sequence", {}) or {}
        payload = {
            "ok": True,
            "function": trim_sentences(" ".join(function_texts)),
            "features": data.get("features", []) or [],
            "length": sequence.get("length"),
        }
        cache[uid] = payload
        return uid, payload
    except Exception:
        payload = {"ok": False, "function": "Function unknown", "features": [], "length": None}
        cache[uid] = payload
        return uid, payload


def feature_start_end(feature: dict[str, Any]) -> tuple[int | None, int | None]:
    location = feature.get("location", {}) or {}
    start = location.get("start", {}).get("value")
    end = location.get("end", {}).get("value")
    if end is None:
        end = start
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None, None


def find_functional_site(features: list[dict[str, Any]], position: int | None) -> str:
    if position is None:
        return NO_SITE_PLACEHOLDER

    hits: list[tuple[int, str, str, int, int]] = []
    for feature in features:
        feature_type = feature.get("type", "")
        if feature_type not in SITE_PRIORITY:
            continue
        start, end = feature_start_end(feature)
        if start is None or end is None:
            continue
        if start <= position <= end:
            description = str(feature.get("description", "") or "").strip()
            hits.append((SITE_PRIORITY.index(feature_type), feature_type, description, start, end))

    if not hits:
        return NO_SITE_PLACEHOLDER

    hits.sort(key=lambda item: item[0])
    _, feature_type, description, start, end = hits[0]
    if description:
        return f"{feature_type} ({description})"
    if start == end:
        return f"{feature_type} at position {start}"
    return f"{feature_type} (positions {start}-{end})"


def clean_loftee_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in LOFTEE_MISSING else text


def interpret_loftee_nmd(row: pd.Series) -> str:
    """根据 LOFTEE 输出解释 NMD/截短可信度，不再自行规则预测。

    LOFTEE 的核心输出是 LoF 高/低可信，而不是实验级 NMD 真值。
    因此这里只在 END_TRUNC 等 LOFTEE 过滤信息出现时标记 NMD escape；
    其他高可信截短写成 Likely_NMD_or_early_truncation，避免过度断言。
    """
    lof = clean_loftee_value(row.get("LoF"))
    lof_filter = clean_loftee_value(row.get("LoF_filter"))
    lof_info = clean_loftee_value(row.get("LoF_info"))
    terms = consequence_terms(row.get("Consequence"))

    if not lof:
        return "Unknown"

    combined = f"{lof_filter};{lof_info}".upper()
    if "END_TRUNC" in combined:
        return "Likely_NMD_escape_or_terminal_truncation"

    if lof.upper() == "HC" and (terms & TRUNCATING_TERMS):
        return "Likely_NMD_or_early_truncation"

    if lof.upper() == "HC":
        return "LOFTEE_high_confidence_LoF"

    if lof.upper() == "LC":
        return "LOFTEE_low_confidence_LoF"

    return "Unknown"


def truncation_percent(row: pd.Series, protein_length: int | None) -> float:
    if protein_length is None or protein_length <= 0:
        return np.nan
    terms = consequence_terms(row.get("Consequence"))
    if not (terms & TRUNCATING_TERMS):
        return np.nan
    position = extract_position(row.get("Protein_position"))
    if position is None or position < 1 or position > protein_length:
        return np.nan
    removed = max(protein_length - position + 1, 0)
    return round(removed / protein_length * 100.0, 2)


def main() -> None:
    print("启动第 06b 步：补充推理链上下文特征。", flush=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    mapping = load_uniprot_mapping()
    cache = load_json(CACHE_JSON)

    symbols = sorted({normalize_symbol(value) for value in df["SYMBOL"].dropna()})
    uids = sorted({mapping[symbol] for symbol in symbols if symbol in mapping})
    missing_uids = [uid for uid in uids if uid not in cache]
    print(f"记录数: {len(df)}", flush=True)
    print(f"唯一基因: {len(symbols)}；UniProt ID: {len(uids)}；需新查询: {len(missing_uids)}", flush=True)

    if missing_uids:
        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_uniprot_context, uid, cache): uid for uid in missing_uids}
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(missing_uids):
                    print(f"UniProt 上下文查询进度: {completed}/{len(missing_uids)}", flush=True)
                time.sleep(0.02)
        save_json(CACHE_JSON, cache)

    gene_functions: list[str] = []
    functional_sites: list[str] = []
    nmd_predictions: list[str] = []
    truncation_values: list[float] = []

    for _, row in df.iterrows():
        symbol = normalize_symbol(row.get("SYMBOL"))
        uid = mapping.get(symbol)
        context = cache.get(uid, {}) if uid else {}
        protein_length = context.get("length")
        features = context.get("features", []) or []

        gene_functions.append(context.get("function") or "Function unknown")
        functional_sites.append(find_functional_site(features, extract_position(row.get("Protein_position"))))
        nmd_predictions.append(interpret_loftee_nmd(row))
        truncation_values.append(truncation_percent(row, protein_length))

    df["Gene_Normal_Function"] = gene_functions
    df["Functional_Site"] = functional_sites
    df["NMD_predicted"] = nmd_predictions
    df["protein_truncation_percent"] = truncation_values

    df.to_csv(OUTPUT_CSV, index=False)

    gene_function_map = (
        df[["SYMBOL", "Gene_Normal_Function"]]
        .dropna(subset=["SYMBOL"])
        .drop_duplicates(subset=["SYMBOL"], keep="first")
        .copy()
    )
    gene_function_map.to_csv(GENE_FUNCTION_MAP_CSV, index=False)

    print("第 06b 步完成。新增特征统计:", flush=True)
    print(f"  Gene_Normal_Function known: {(df['Gene_Normal_Function'] != 'Function unknown').sum()}/{len(df)}", flush=True)
    print(f"  Functional_Site hit: {(df['Functional_Site'] != NO_SITE_PLACEHOLDER).sum()}/{len(df)}", flush=True)
    print("  NMD_predicted:", flush=True)
    print(df["NMD_predicted"].value_counts(dropna=False).to_string(), flush=True)
    print(f"  protein_truncation_percent non-null: {df['protein_truncation_percent'].notna().sum()}/{len(df)}", flush=True)
    print(f"输出文件: {OUTPUT_CSV}", flush=True)
    print(f"基因功能映射: {GENE_FUNCTION_MAP_CSV}", flush=True)


if __name__ == "__main__":
    main()
