"""Apply every compression scheme x calibration source to stored embeddings and score retrieval.

Usage: python src/evaluate.py --model me5-small [--calib_n 1388] [--seed 0]
Output: results/<model>.csv  (one row per lang x scheme x calibration x query_mode)

Calibration sources for int8/int4/int2/binary_cal (all with calib_n vectors unless noted):
  english   : English docs+queries of this model (what practitioners typically do)
  target    : the target language's own docs+queries (in-domain upper bound; leaks the eval set)
  target64  : only 64 target-language texts (cheap, realistic "few-shot" calibration)
  mixed     : sampled uniformly across all OTHER languages (leave-target-out, no leakage)
  mixed_lat : sampled uniformly across the other Latin-script languages only
Query modes:
  asym : corpus compressed, query kept float32 (on-device storage scenario)
  sym  : corpus and query compressed the same way
"""
from __future__ import annotations
import argparse, csv, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from quant import SCHEMES, BITS, calib_ranges, clip_fraction
from metrics import retrieval_metrics
from data import script_of, english_code, WIKI_CODE


def load(model, lang, task="belebele"):
    d = os.path.join("embeddings", model) if task == "belebele" else os.path.join("embeddings", task, model)
    z = np.load(os.path.join(d, f"{lang}.npz"))
    return z["docs"], z["queries"], z["qrels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="belebele", choices=["belebele", "wikimulti"])
    ap.add_argument("--calib_n", type=int, default=1388)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    edir = f"embeddings/{a.model}" if a.task == "belebele" else f"embeddings/{a.task}/{a.model}"
    langs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{edir}/*.npz"))
    ENG = english_code(a.task)
    assert ENG in langs, "need English embeddings for the english calibration"
    emb = {l: load(a.model, l, a.task) for l in langs}
    stack = lambda l: np.concatenate([emb[l][0], emb[l][1]])

    def sample(X, n):
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        return X[idx]

    calib = {"english": sample(stack(ENG), a.calib_n)}
    pool = {l: sample(stack(l), max(1, a.calib_n // (len(langs) - 1)) + 1) for l in langs}
    code = (lambda l: l) if a.task == "belebele" else (lambda l: WIKI_CODE[l])
    latin = [l for l in langs if code(l).endswith("_Latn")]
    pool_lat = {l: sample(stack(l), max(1, a.calib_n // max(1, len(latin) - 1)) + 1) for l in latin}

    rows = []
    for lang in langs:
        D, Q, qrels = emb[lang]
        calib["target"] = sample(stack(lang), a.calib_n)
        calib["target64"] = sample(stack(lang), 64)
        calib["mixed"] = np.concatenate([pool[l] for l in langs if l != lang])[: a.calib_n]
        others_lat = [pool_lat[l] for l in latin if l != lang]
        calib["mixed_lat"] = np.concatenate(others_lat)[: a.calib_n] if others_lat else calib["mixed"]
        S0 = Q @ D.T
        base = retrieval_metrics(S0, qrels)
        # geometry diagnostics: how much margin does fp32 leave for quantisation noise?
        rel = S0[np.arange(len(qrels)), qrels]
        S_masked = S0.copy(); S_masked[np.arange(len(qrels)), qrels] = -np.inf
        margin = rel - S_masked.max(1)                       # >0 iff relevant doc ranked first
        # off-diagonal doc-doc cosine stats without materialising the NxN gram matrix (memory-safe)
        n = len(D); npairs = n * (n - 1) / 2
        s1 = (D.sum(0) @ D.sum(0) - (D * D).sum()) / 2          # sum over i<j of D_i.D_j
        s2 = (np.linalg.norm(D.T @ D) ** 2 - ((D * D).sum(1) ** 2).sum()) / 2   # sum over i<j of (D_i.D_j)^2
        cos_mean = s1 / npairs; cos_std = np.sqrt(max(s2 / npairs - cos_mean ** 2, 0.0))
        diag = dict(margin_mean=float(margin.mean()), margin_p25=float(np.percentile(margin, 25)),
                    doc_cos_mean=float(cos_mean), doc_cos_std=float(cos_std),
                    q_norm_var=float(Q.var(0).sum()), dim_var_cv=float(D.var(0).std() / D.var(0).mean()))
        for scheme, (fn, needs_cal) in SCHEMES.items():
            for cal_name in (["english", "target", "target64", "mixed", "mixed_lat"] if needs_cal else ["none"]):
                C = calib.get(cal_name)
                Dq = fn(D, C)
                for qmode in ("asym", "sym"):
                    Qq = Q if qmode == "asym" else fn(Q, C)
                    m = retrieval_metrics(Qq @ Dq.T, qrels)
                    extra = {}
                    if scheme.startswith("int"):
                        extra["clip_frac"] = clip_fraction(D, calib_ranges(C))
                    rows.append(dict(model=a.model, task=a.task, lang=code(lang), script=script_of(code(lang)), scheme=scheme,
                                     bits=BITS[scheme], calib=cal_name, qmode=qmode, **m,
                                     base_ndcg=base["ndcg_at_10"], **diag,
                                     rel_drop=(base["ndcg_at_10"] - m["ndcg_at_10"]) / max(base["ndcg_at_10"], 1e-9),
                                     **extra))
        print(lang, f"base={base['ndcg_at_10']:.3f}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = f"results/{a.model}.csv" if a.task == "belebele" else f"results/{a.task}_{a.model}.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
