"""Belebele retrieval data, built exactly like MTEB's BelebeleRetrieval task.

Each language: 488 FLORES passages (corpus) and 900 questions (queries),
one relevant passage per question. All 122 languages share the same
passages/questions (translations), which makes cross-language comparison clean.
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from datasets import load_dataset

HF_PATH = "mteb/belebele"
HF_REV = "979a211276faa22f671e69d096634193567cfd05"


@lru_cache(maxsize=1)
def all_languages() -> list[str]:
    """The 122 monolingual subsets of MTEB BelebeleRetrieval (from the shipped data if present, else from mteb)."""
    local = os.path.join(BELEBELE_RAW, "data")
    if os.path.isdir(local) and len(os.listdir(local)) >= 122:
        return sorted(f[:-6] for f in os.listdir(local) if f.endswith(".jsonl"))
    import mteb
    langs = mteb.get_task("BelebeleRetrieval").metadata.eval_langs
    return sorted(k.split("-")[0] for k in langs if k.split("-")[0] == k.split("-")[1])


@dataclass
class LangData:
    lang: str
    doc_ids: list[str]
    docs: list[str]
    query_ids: list[str]
    queries: list[str]
    qrels: np.ndarray  # (n_queries,) index into docs of the relevant passage


BELEBELE_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "belebele_raw")


def _belebele_rows(lang: str):
    """Rows of the test split, from the parquet shipped in data/belebele_raw if present, else from the Hub."""
    import json
    local = os.path.join(BELEBELE_RAW, "data", f"{lang}.jsonl")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return load_dataset(HF_PATH, lang, split="test", revision=HF_REV)


def load_lang(lang: str) -> LangData:
    ds = _belebele_rows(lang)
    link_to_idx, docs = {}, []
    for row in ds:
        if row["link"] not in link_to_idx:
            link_to_idx[row["link"]] = len(docs)
            docs.append(row["flores_passage"])
    q_to_idx, queries, qrels = {}, [], []
    for row in ds:
        q = row["question"]
        if q not in q_to_idx:
            q_to_idx[q] = len(queries)
            queries.append(q)
            qrels.append(link_to_idx[row["link"]])
    return LangData(
        lang=lang,
        doc_ids=[f"C{i}" for i in range(len(docs))],
        docs=docs,
        query_ids=[f"Q{i}" for i in range(len(queries))],
        queries=queries,
        qrels=np.asarray(qrels),
    )


def script_of(lang: str) -> str:
    return lang.split("_")[1]


# ---------------------------------------------------------------------------
# Second task: WikipediaRetrievalMultilingual (MTEB) -- 16 languages, ~13.5k passages, 1500 queries each
# ---------------------------------------------------------------------------
TASKS = {"belebele": "BelebeleRetrieval", "wikimulti": "WikipediaRetrievalMultilingual"}
WIKI_LANGS = ["bg", "bn", "cs", "da", "de", "en", "fa", "fi", "hi", "it", "nl", "pt", "ro", "sr", "no", "sv"]
WIKI_CODE = {"bg": "bul_Cyrl", "bn": "ben_Beng", "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "en": "eng_Latn",
             "fa": "pes_Arab", "fi": "fin_Latn", "hi": "hin_Deva", "it": "ita_Latn", "nl": "nld_Latn", "pt": "por_Latn",
             "ro": "ron_Latn", "sr": "srp_Cyrl", "no": "nob_Latn", "sv": "swe_Latn"}


def task_languages(task: str) -> list[str]:
    return all_languages() if task == "belebele" else WIKI_LANGS


def english_code(task: str) -> str:
    return "eng_Latn" if task == "belebele" else "en"


WIKI_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wikimulti_raw")


def _load_wiki_local(lang: str) -> LangData:
    """Read the MTEB parquet files shipped in data/wikimulti_raw (no Hub API calls)."""
    import pandas as pd
    corpus = pd.read_parquet(os.path.join(WIKI_RAW, f"{lang}-corpus", "test-00000-of-00001.parquet"))
    queries = pd.read_parquet(os.path.join(WIKI_RAW, f"{lang}-queries", "test-00000-of-00001.parquet"))
    qrels = pd.read_parquet(os.path.join(WIKI_RAW, f"{lang}-qrels", "test-00000-of-00001.parquet"))
    doc_ids = corpus["_id"].tolist() if "_id" in corpus else corpus["id"].tolist()
    docs = corpus["text"].tolist()
    idx = {i: k for k, i in enumerate(doc_ids)}
    qid_col = "_id" if "_id" in queries else "id"
    qtext = dict(zip(queries[qid_col], queries["text"]))
    best = {}
    for qid, did, sc in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
        if qid not in best or sc > best[qid][1]:
            best[qid] = (did, sc)
    q_ids = [q for q in queries[qid_col] if q in best]
    return LangData(lang=lang, doc_ids=doc_ids, docs=docs, query_ids=q_ids, queries=[qtext[q] for q in q_ids],
                    qrels=np.asarray([idx[best[q][0]] for q in q_ids]))


def load_task_lang(task: str, lang: str) -> LangData:
    if task == "belebele":
        return load_lang(lang)
    if os.path.isdir(os.path.join(WIKI_RAW, f"{lang}-corpus")):
        return _load_wiki_local(lang)
    import mteb
    t = mteb.get_task(TASKS[task], eval_splits=["test"], hf_subsets=[lang])
    t.load_data()
    d = t.dataset[lang]["test"]
    corpus, queries, rel = d["corpus"], d["queries"], d["relevant_docs"]
    doc_ids = list(corpus["id"]); docs = list(corpus["text"])
    idx = {i: k for k, i in enumerate(doc_ids)}
    q_ids, qs, qrels = [], [], []
    for qid, qt in zip(queries["id"], queries["text"]):
        r = rel.get(qid, {})
        if not r:
            continue
        q_ids.append(qid); qs.append(qt); qrels.append(idx[max(r, key=r.get)])
    return LangData(lang=lang, doc_ids=doc_ids, docs=docs, query_ids=q_ids, queries=qs, qrels=np.asarray(qrels))
