#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 02 步：构建 UniProt 映射并下载 AlphaFold PDB。

输入：
  data/processed/01_BIOREASON_with_DBs.csv

输出：
  data/processed/uniprot_mapping.csv
  data/processed/alphafold_pdbs/*.pdb

说明：
  1. UniProt 查询限定 human + reviewed，尽量对应人工审核的经典蛋白。
  2. 如果已有 uniprot_mapping.csv，会作为缓存复用，只查询新增基因。
  3. AlphaFold PDB 已存在时直接跳过，便于断点续跑。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "processed", "01_BIOREASON_with_DBs.csv")
MAPPING_CSV = os.path.join(BASE_DIR, "processed", "uniprot_mapping.csv")
PDB_DIR = os.path.join(BASE_DIR, "processed", "alphafold_pdbs")
os.makedirs(PDB_DIR, exist_ok=True)

MAX_WORKERS = 8
UNIPROT_TIMEOUT = 12
ALPHAFOLD_TIMEOUT = 20
REQUEST_HEADERS = {
    "User-Agent": "gof-bioreason-dataset-builder/1.0"
}


def read_target_genes() -> list[str]:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    genes = sorted({
        str(gene).strip()
        for gene in df["SYMBOL"].dropna()
        if str(gene).strip() and str(gene).strip().lower() not in {"none", "nan", "-"}
    })
    print(f"读取 01 大表完成，需要映射的唯一基因数: {len(genes)}", flush=True)
    return genes


def load_mapping_cache() -> dict[str, str]:
    if not os.path.exists(MAPPING_CSV):
        return {}
    cache_df = pd.read_csv(MAPPING_CSV, low_memory=False)
    if not {"SYMBOL", "UniProt_ID"}.issubset(cache_df.columns):
        return {}
    cache: dict[str, str] = {}
    for row in cache_df.itertuples(index=False):
        symbol = str(row.SYMBOL).strip()
        uid = str(row.UniProt_ID).strip()
        if symbol and uid and uid.lower() not in {"nan", "none", "-"}:
            cache[symbol] = uid
    print(f"读取已有 UniProt 缓存: {len(cache)} 个基因", flush=True)
    return cache


def query_uniprot(gene: str) -> tuple[str, str]:
    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query=(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)"
        "&fields=accession&size=1"
    )
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=UNIPROT_TIMEOUT, verify=False)
        if response.status_code != 200:
            return gene, "nan"
        data = response.json()
        results = data.get("results") or []
        if not results:
            return gene, "nan"
        return gene, str(results[0].get("primaryAccession") or "nan")
    except Exception:
        return gene, "nan"


def build_uniprot_mapping(genes: list[str]) -> pd.DataFrame:
    cache = load_mapping_cache()
    missing_genes = [gene for gene in genes if gene not in cache]
    print(f"需要新查询 UniProt 的基因数: {len(missing_genes)}", flush=True)

    mapping = dict(cache)
    if missing_genes:
        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(query_uniprot, gene): gene for gene in missing_genes}
            for future in as_completed(futures):
                gene, uid = future.result()
                mapping[gene] = uid
                completed += 1
                if completed % 50 == 0 or completed == len(missing_genes):
                    print(f"UniProt 查询进度: {completed}/{len(missing_genes)}", flush=True)
                time.sleep(0.02)

    mapping_df = pd.DataFrame(
        [{"SYMBOL": gene, "UniProt_ID": mapping.get(gene, "nan")} for gene in genes]
    )
    mapping_df.to_csv(MAPPING_CSV, index=False)
    matched = int(mapping_df["UniProt_ID"].astype(str).str.lower().ne("nan").sum())
    print(f"UniProt 映射完成: 成功 {matched}/{len(mapping_df)}", flush=True)
    return mapping_df


def download_one_pdb(uid_raw: str) -> tuple[str, str]:
    uid = str(uid_raw).strip()
    if not uid or uid.lower() in {"nan", "none", "-"}:
        return uid, "skip_no_uniprot"

    try:
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
        response = requests.get(api_url, headers=REQUEST_HEADERS, timeout=ALPHAFOLD_TIMEOUT, verify=False)
        if response.status_code == 404:
            return uid, "not_in_alphafold"
        if response.status_code != 200:
            return uid, f"api_http_{response.status_code}"

        data = response.json()
        if not isinstance(data, list) or not data:
            return uid, "not_in_alphafold"
        pdb_url = data[0].get("pdbUrl")
        if not pdb_url:
            return uid, "no_pdb_url"

        file_name = pdb_url.split("/")[-1]
        save_path = os.path.join(PDB_DIR, file_name)
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return uid, "exists"

        pdb_response = requests.get(pdb_url, headers=REQUEST_HEADERS, timeout=ALPHAFOLD_TIMEOUT, verify=False)
        if pdb_response.status_code != 200:
            return uid, f"pdb_http_{pdb_response.status_code}"
        with open(save_path, "wb") as handle:
            handle.write(pdb_response.content)
        return uid, "downloaded"
    except Exception:
        return uid, "error"


def download_alphafold_pdbs(mapping_df: pd.DataFrame) -> None:
    uids = sorted({
        str(uid).strip()
        for uid in mapping_df["UniProt_ID"].dropna()
        if str(uid).strip().lower() not in {"nan", "none", "-"}
    })
    print(f"准备检查/下载 AlphaFold PDB: {len(uids)} 个 UniProt ID", flush=True)

    status_counts: dict[str, int] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one_pdb, uid): uid for uid in uids}
        for future in as_completed(futures):
            _, status = future.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            completed += 1
            if completed % 50 == 0 or completed == len(uids):
                print(f"AlphaFold PDB 进度: {completed}/{len(uids)}", flush=True)
            time.sleep(0.02)

    print("AlphaFold PDB 状态统计:", flush=True)
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}", flush=True)


def main() -> None:
    print("启动第 02 步：UniProt 映射与 AlphaFold PDB 下载。", flush=True)
    genes = read_target_genes()
    mapping_df = build_uniprot_mapping(genes)
    download_alphafold_pdbs(mapping_df)
    print("第 02 步完成。", flush=True)


if __name__ == "__main__":
    main()
