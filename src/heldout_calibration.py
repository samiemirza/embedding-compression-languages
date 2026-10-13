"""Calibration without content overlap (E5-small, Belebele, 122 languages).

The Belebele English calibration set is the English evaluation data, and every Belebele calibration set contains
translations of the evaluated passages. Here the calibration set is 1388 English Wikipedia passages from the
MTEB multilingual Wikipedia retrieval corpus (different content and domain), five random draws. We also redraw
the mixed (leave-target-out) Belebele sample with five seeds.
Output: results/heldout_calibration_me5-small.csv (per language x calibration x seed x scheme)
"""
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from metrics import retrieval_metrics
from models import MODELS
from data import load_task_lang

MODEL, CAL_N = "me5-small", 1388
hf_id, qp, dp, trc, *rest = MODELS[MODEL]
langs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"embeddings/{MODEL}/*.npz"))
E = {l: np.load(f"embeddings/{MODEL}/{l}.npz") for l in langs}
nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)

cache = f"embeddings/{MODEL}_wiki_en_calib.npy"
if not os.path.exists(cache):
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(hf_id, device=dev); m.max_seq_length = 512
    docs = load_task_lang("wikimulti", "en").docs
    pick = np.random.default_rng(0).choice(len(docs), size=5 * CAL_N, replace=False)
    X = m.encode([dp + docs[i] for i in pick], batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(cache, X)
W = np.load(cache)

def ranges(C, pct=0.0):
    return (C.min(0), C.max(0)) if pct <= 0 else (np.percentile(C, pct, axis=0), np.percentile(C, 100 - pct, axis=0))
def uniform(X, r, bits):
    lo, hi = r; lv = 2 ** bits - 1; st = np.where((hi - lo) / lv == 0, 1e-12, (hi - lo) / lv)
    return (np.clip(np.round((X - lo) / st), 0, lv) * st + lo).astype(np.float32)
sign = lambda X, t=0.0: np.where(X > t, 1.0, -1.0).astype(np.float32)

stack = lambda l: np.concatenate([E[l]["docs"], E[l]["queries"]])
cals = {}
for s in range(5):
    cals[("wiki_en", s)] = W[s * CAL_N:(s + 1) * CAL_N]
cals[("bele_en", 0)] = stack("eng_Latn")
rows = []
for li, l in enumerate(langs):
    D, Q, qr = E[l]["docs"], E[l]["queries"], E[l]["qrels"]
    base = retrieval_metrics(Q @ D.T, qr)["ndcg_at_10"]
    mixed = {}
    for s in range(5):
        r = np.random.default_rng(1000 + s)
        others = [x for x in langs if x != l]
        per = CAL_N // len(others) + 1
        mixed[("mixed", s)] = np.concatenate([stack(x)[r.choice(len(stack(x)), per, replace=False)] for x in others])[:CAL_N]
    for (cal, s), C in list(cals.items()) + list(mixed.items()):
        mu = C.mean(0)
        res = {"int2": Q @ uniform(D, ranges(C), 2).T, "int2_clip": Q @ uniform(D, ranges(C, 5.0), 2).T,
               "bin_mean_thr": Q @ sign(D, mu).T}
        Dt, Qt = nrm(D - mu), nrm(Q - mu); bt = retrieval_metrics(Qt @ Dt.T, qr)["ndcg_at_10"]
        res["centered_sign"] = Qt @ sign(Dt).T
        for sc, Sc in res.items():
            n = retrieval_metrics(Sc, qr)["ndcg_at_10"]
            rows.append(dict(lang=l, calib=cal, seed=s, scheme=sc, base_ndcg=base, ndcg=n, loss=100 * (base - n) / base,
                             loss_vs_t=(100 * (bt - n) / bt) if sc == "centered_sign" else np.nan))
    if li % 20 == 0: print(li, l, flush=True)
pd.DataFrame(rows).to_csv(f"results/heldout_calibration_{MODEL}.csv", index=False); print("done")
