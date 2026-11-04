"""Equal-storage and centering experiments, on re-computed embeddings.

Run per model once its embeddings exist:  python src/extra_experiments.py --task belebele --model me5-small

1. Sanity check: float32 nDCG@10 and sign-binary loss per language against the stored results.
2. Matched-byte baselines for the 1-bit (d/8 bytes) and 2-bit (d/4 bytes) budgets:
   - pq:      product quantization, 8-bit codes, d/8 or d/4 sub-quantizers (faiss), query float32
   - pq_c:    the same after centering on the calibration mean
   - pca8:    PCA to d/8 or d/4 dimensions, stored at int8 (min/max range), query projected in float32
   PQ and PCA are fitted on English (1388 vectors, the calibration set of the paper) or on 10,000 vectors
   sampled uniformly over all languages of the task ("multi10k"; enough to train 256-centroid PQ codebooks;
   each language contributes about 1% of it on Belebele).
3. Centering sample size: binary loss after centering on the mean of n English vectors,
   n in {16, 64, 256, 1388}; five random draws for n < 1388.
4. (task=wikimulti) centering and sign binary on the larger corpus, calibrated on that task's English.

Output: results/extra_<task>_<model>.csv (long format: one row per language x method x budget x calibration x n)
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, pandas as pd
import faiss
sys.path.insert(0, os.path.dirname(__file__))
from data import english_code, WIKI_CODE
from metrics import retrieval_metrics  # quant.py is not imported: it loads torch, whose OpenMP clashes with faiss on macOS


def calib_ranges(C, pct=0.0):
    if pct <= 0:
        return C.min(0), C.max(0)
    return np.percentile(C, pct, axis=0), np.percentile(C, 100 - pct, axis=0)


def uniform(X, ranges, bits):  # identical to quant.uniform
    lo, hi = ranges; levels = 2 ** bits - 1
    step = np.where((hi - lo) / levels == 0, 1e-12, (hi - lo) / levels)
    return (np.clip(np.round((X - lo) / step), 0, levels) * step + lo).astype(np.float32)


def binary(X, thresh=None):  # identical to quant.binary
    return np.where(X > (0.0 if thresh is None else thresh), 1.0, -1.0).astype(np.float32)

CAL_N, SEED = 1388, 0


def ndcg(Q, D, qrels):
    return retrieval_metrics(Q @ D.T, qrels)["ndcg_at_10"]


def margin(S, qrels):
    rel = S[np.arange(len(qrels)), qrels]; M = S.copy(); M[np.arange(len(qrels)), qrels] = -np.inf
    return float((rel - M.max(1)).mean())


def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def pq_fit(C, m):
    pq = faiss.ProductQuantizer(C.shape[1], m, 8)
    pq.train(np.ascontiguousarray(C, dtype=np.float32))
    return lambda X: pq.decode(pq.compute_codes(np.ascontiguousarray(X, dtype=np.float32)))


def pca_fit(C, k):
    mu = C.mean(0); _, _, Vt = np.linalg.svd(C - mu, full_matrices=False); P = Vt[:k].T
    rng_ = calib_ranges((C - mu) @ P)
    return (lambda X: uniform((X - mu) @ P, rng_, 8)), (lambda Q: (Q - mu) @ P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="belebele", choices=["belebele", "wikimulti"])
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    edir = f"embeddings/{a.model}" if a.task == "belebele" else f"embeddings/wikimulti/{a.model}"
    langs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{edir}/*.npz"))
    emb = {l: (lambda z: (z["docs"], z["queries"], z["qrels"]))(np.load(f"{edir}/{l}.npz")) for l in langs}
    d = emb[langs[0]][0].shape[1]
    rng = np.random.default_rng(SEED)
    stack = lambda l: np.concatenate([emb[l][0], emb[l][1]])
    samp = lambda X, n, r=rng: X[r.choice(len(X), size=min(n, len(X)), replace=False)]
    eng_all = stack(english_code(a.task)); cal_eng = samp(eng_all, CAL_N)
    code = (lambda l: l) if a.task == "belebele" else (lambda l: WIKI_CODE[l])  # results files use ISO-script codes
    allv = np.concatenate([stack(l) for l in langs]); cal_multi = samp(allv, 10000); del allv

    stored = pd.read_csv(f"results/{a.model}.csv" if a.task == "belebele" else f"results/wikimulti_{a.model}.csv")
    stored = stored[stored.qmode == "asym"]
    st_base = stored[stored.scheme == "fp32"].set_index("lang").base_ndcg
    st_bin = stored[stored.scheme == "binary"].set_index("lang").ndcg_at_10

    budgets = {"1bit": d // 8, "2bit": d // 4}
    fits = {}
    for calib, C in [("english", cal_eng), ("multi10k", cal_multi)]:
        for b, m in budgets.items():
            fits[("pq", b, calib)] = pq_fit(C, m)
            mu = C.mean(0); fits[("pq_c", b, calib)] = (mu, pq_fit(normalize(C - mu), m))
            fits[("pca8", b, calib)] = pca_fit(C, m)

    rows = []
    def add(lang, method, budget, calib, n, base, nd, extra=None):
        r = dict(task=a.task, model=a.model, lang=code(lang), method=method, budget=budget, calib=calib, n=n,
                 base_ndcg=base, ndcg=nd, loss=100 * (base - nd) / max(base, 1e-9))
        r.update(extra or {}); rows.append(r)

    for i, lang in enumerate(langs):
        D, Q, qrels = emb[lang]
        S0 = Q @ D.T; base = retrieval_metrics(S0, qrels)["ndcg_at_10"]
        nb = ndcg(Q, binary(D), qrels)
        add(lang, "check_fp32", "-", "-", 0, base, base, dict(stored_base=st_base.get(code(lang), np.nan)))
        add(lang, "check_sign", "1bit", "-", 0, base, nb, dict(stored_ndcg=st_bin.get(code(lang), np.nan)))
        for calib in ["english", "multi10k"]:
            for b in budgets:
                add(lang, "pq", b, calib, 0, base, ndcg(Q, fits[("pq", b, calib)](D), qrels))
                mu, fc = fits[("pq_c", b, calib)]
                Dt, Qt = normalize(D - mu), normalize(Q - mu)
                add(lang, "pq_c", b, calib, 0, base, ndcg(Qt, fc(Dt), qrels))
                fd, fq = fits[("pca8", b, calib)]
                add(lang, "pca8", b, calib, 0, base, ndcg(fq(Q), fd(D), qrels))
        # centering sample size (English), binary after centering
        for n in [16, 64, 256, CAL_N]:
            for s in range(1 if n == CAL_N else 5):
                C = cal_eng if n == CAL_N else samp(eng_all, n, np.random.default_rng(100 * n + s))
                mu = C.mean(0); Dt, Qt = normalize(D - mu), normalize(Q - mu)
                St = Qt @ Dt.T; bt = retrieval_metrics(St, qrels)["ndcg_at_10"]
                nd = ndcg(Qt, binary(Dt), qrels)
                add(lang, "center_sign", "1bit", "english", n, base, nd,
                    dict(seed=s, base_ndcg_t=bt, loss_vs_t=100 * (bt - nd) / max(bt, 1e-9), margin_t=margin(St, qrels)))
        if a.task == "wikimulti":
            add(lang, "sign_margin", "-", "-", 0, base, base, dict(margin=margin(S0, qrels)))
        print(f"[{i+1}/{len(langs)}] {lang} base {base:.3f} (stored {st_base.get(code(lang), np.nan):.3f})", flush=True)

    out = f"results/extra_{a.task}_{a.model}.csv"
    pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out)


if __name__ == "__main__":
    main()
