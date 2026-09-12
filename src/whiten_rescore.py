"""Two extra experiments on stored Belebele embeddings (all models, 122 languages).

1. Whitening: does removing the common component / whitening the space (fitted on a calibration set)
   raise the margin and reduce the compression loss of anisotropic models?
   Variants: none | center (subtract calibration mean) | zca (ZCA-whiten with shrinkage), then re-normalise.
   Calibration: english (1388 English vectors) or mixed (leave-target-out, 1388 vectors).
   Reported per language: fp32 nDCG after the transform, margin after the transform, and loss of
   int4 / int2-clip / binary relative to the *transformed* fp32 and to the *original* fp32.

2. Rescoring: retrieve top-k with the compressed corpus (binary or int2-clip, query float32), then
   rescore the k candidates with float32 query x int8 corpus (realistic) or x float32 corpus (upper bound).
   k in {10, 20, 50, 100}. k=10 equals no rescoring.

Output: results/whitening.csv, results/rescoring.csv
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from quant import uniform, calib_ranges, binary
from metrics import retrieval_metrics

MODELS = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
CAL_N, SEED = 1388, 0


def load(model, lang):
    z = np.load(f"embeddings/{model}/{lang}.npz"); return z["docs"], z["queries"], z["qrels"]


def margin(S, qrels):
    rel = S[np.arange(len(qrels)), qrels]; M = S.copy(); M[np.arange(len(qrels)), qrels] = -np.inf
    return float((rel - M.max(1)).mean())


def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def fit_transform(C, kind, shrink=0.1):
    mu = C.mean(0)
    if kind == "center":
        return lambda X: normalize(X - mu)
    Xc = C - mu
    cov = Xc.T @ Xc / len(Xc)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    w, V = np.linalg.eigh(cov)
    W = V @ np.diag(1 / np.sqrt(w)) @ V.T
    return lambda X: normalize((X - mu) @ W)


def main():
    rng = np.random.default_rng(SEED)
    wrows, rrows = [], []
    for model in MODELS:
        langs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"embeddings/{model}/*.npz"))
        emb = {l: load(model, l) for l in langs}
        stack = lambda l: np.concatenate([emb[l][0], emb[l][1]])
        samp = lambda X, n: X[rng.choice(len(X), size=min(n, len(X)), replace=False)]
        cal_eng = samp(stack("eng_Latn"), CAL_N)
        pool = {l: samp(stack(l), CAL_N // (len(langs) - 1) + 1) for l in langs}
        print(model, len(langs), "languages", flush=True)
        for lang in langs:
            D, Q, qrels = emb[lang]
            S0 = Q @ D.T; base = retrieval_metrics(S0, qrels)["ndcg_at_10"]
            cal_mix = np.concatenate([pool[l] for l in langs if l != lang])[:CAL_N]
            # ---- 1. whitening ----
            for cal_name, C in [("english", cal_eng), ("mixed", cal_mix)]:
                for kind in ["none", "center", "zca"]:
                    if kind == "none" and cal_name == "mixed":
                        continue
                    T = (lambda X: X) if kind == "none" else fit_transform(C, kind)
                    Dt, Qt, Ct = T(D), T(Q), T(C)
                    St = Qt @ Dt.T; b_t = retrieval_metrics(St, qrels)["ndcg_at_10"]
                    G = Dt @ Dt.T; iu = np.triu_indices(len(Dt), 1)
                    row = dict(model=model, lang=lang, calib=cal_name, transform=kind, base_ndcg=base, base_ndcg_t=b_t,
                               margin_t=margin(St, qrels), doc_cos_t=float(G[iu].mean()))
                    for sc, fn in [("int4", lambda X: uniform(X, calib_ranges(Ct), 4)),
                                   ("int2_p5", lambda X: uniform(X, calib_ranges(Ct, 5.0), 2)),
                                   ("binary", lambda X: binary(X))]:
                        n = retrieval_metrics(Qt @ fn(Dt).T, qrels)["ndcg_at_10"]
                        row[f"{sc}_ndcg"] = n
                        row[f"{sc}_loss_vs_t"] = 100 * (b_t - n) / max(b_t, 1e-9)
                        row[f"{sc}_loss_vs_orig"] = 100 * (base - n) / max(base, 1e-9)
                    wrows.append(row)
            # ---- 2. rescoring ----
            D8 = uniform(D, calib_ranges(cal_eng), 8)
            for sc, Dc in [("binary", binary(D)), ("int2_p5", uniform(D, calib_ranges(cal_eng, 5.0), 2))]:
                Sc = Q @ Dc.T
                for k in [10, 20, 50, 100]:
                    top = np.argpartition(-Sc, k - 1, axis=1)[:, :k]
                    for rname, Dr in [("int8", D8), ("fp32", D)]:
                        Sr = np.full_like(Sc, -np.inf)
                        rs = np.einsum("qd,qkd->qk", Q, Dr[top])
                        np.put_along_axis(Sr, top, rs, axis=1)
                        n = retrieval_metrics(Sr, qrels)["ndcg_at_10"]
                        rrows.append(dict(model=model, lang=lang, scheme=sc, k=k, rescore=rname, base_ndcg=base,
                                          ndcg=n, loss=100 * (base - n) / max(base, 1e-9)))
    pd.DataFrame(wrows).to_csv("results/whitening.csv", index=False)
    pd.DataFrame(rrows).to_csv("results/rescoring.csv", index=False)
    print("wrote results/whitening.csv, results/rescoring.csv")


if __name__ == "__main__":
    main()
