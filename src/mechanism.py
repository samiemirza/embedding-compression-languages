"""Mechanism checks on stored E5-small Belebele embeddings (all 122 languages).

1. Score perturbation: for sign binarization (query float32), how much does compression perturb the score
   gap between the relevant document and its strongest float32 distractor, relative to the float32 margin,
   and is a ranking flip predicted by perturbation > margin?
2. Graded centering: subtract alpha * mu (mu = English calibration mean) from documents and queries,
   alpha in {0, .25, .5, .75, 1}; report anisotropy (mean passage cosine), margin, score noise and binary loss.

Scores are compared after scaling compressed scores to the float32 scale per query (least squares),
so that perturbation is measured in the units of the margin.
Also: bits shared between documents and the query-independent score offset, and centering one side only.
Output: results/mechanism_me5-small.csv (per language x alpha), mechanism_queries_me5-small.csv,
mechanism_bits_me5-small.csv, mechanism_onesided_me5-small.csv
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from metrics import retrieval_metrics

MODEL = "me5-small"
langs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"embeddings/{MODEL}/*.npz"))
E = {l: np.load(f"embeddings/{MODEL}/{l}.npz") for l in langs}
eng = E["eng_Latn"]; mu = np.concatenate([eng["docs"], eng["queries"]]).mean(0)
nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
rows = []
for l in langs:
    D0, Q0, qr = E[l]["docs"], E[l]["queries"], E[l]["qrels"]
    base0 = retrieval_metrics(Q0 @ D0.T, qr)["ndcg_at_10"]
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        D, Q = nrm(D0 - alpha * mu), nrm(Q0 - alpha * mu)
        S = Q @ D.T; n = len(qr); idx = np.arange(n)
        rel = S[idx, qr]; M = S.copy(); M[idx, qr] = -np.inf; j = M.argmax(1); margin = rel - M[idx, j]
        B = np.where(D > 0, 1.0, -1.0); Sb = Q @ B.T
        # put compressed scores on the float32 scale per query: Sb ~ a*S + c (least squares over documents)
        Sc = S - S.mean(1, keepdims=True); Sbc = Sb - Sb.mean(1, keepdims=True)
        a = (Sc * Sbc).sum(1) / (Sc * Sc).sum(1)
        Sb_scaled = Sbc / a[:, None] + S.mean(1, keepdims=True)
        err = Sb_scaled - S                                   # score perturbation per (query, doc)
        noise = err.std(1).mean()                             # typical perturbation of a single score
        gap_change = err[idx, qr] - err[idx, j]               # change of the relevant-minus-top-distractor gap
        flip_pred = (margin > 0) & (-gap_change > margin)     # perturbation exceeds a positive margin
        rank0 = (S > rel[:, None]).sum(1) + 1
        relb = Sb[idx, qr]; rankb = (Sb > relb[:, None]).sum(1) + 1
        lost_top1 = (rank0 == 1) & (rankb > 1)
        G = D @ D.T; iu = np.triu_indices(len(D), 1)
        nb = retrieval_metrics(Sb, qr)["ndcg_at_10"]; nt = retrieval_metrics(S, qr)["ndcg_at_10"]
        rows.append(dict(model=MODEL, lang=l, alpha=alpha, doc_cos=float(G[iu].mean()), margin=float(margin.mean()),
                         noise=float(noise), noise_over_margin=float(noise / max(np.median(np.abs(margin)), 1e-9)),
                         ndcg_fp32=nt, ndcg_bin=nb, loss_vs_orig=100 * (base0 - nb) / base0, loss_vs_t=100 * (nt - nb) / nt,
                         top1_lost=float(lost_top1[rank0 == 1].mean()) if (rank0 == 1).any() else np.nan,
                         flip_pred_rate=float(flip_pred[rank0 == 1].mean()) if (rank0 == 1).any() else np.nan,
                         flip_agree=float(((flip_pred == lost_top1)[rank0 == 1]).mean()) if (rank0 == 1).any() else np.nan))
pd.DataFrame(rows).to_csv(f"results/mechanism_{MODEL}.csv", index=False); print("wrote", len(rows), "rows")

# 3. Per query: probability that a float32 top-1 query loses top-1 under sign binarization, by its margin
#    relative to the perturbation of its own scores (std over documents of the scaled compressed-minus-float32 score).
qrows = []
for l in langs:
    D0, Q0, qr = E[l]["docs"], E[l]["queries"], E[l]["qrels"]; idx = np.arange(len(qr))
    for alpha in [0.0, 1.0]:
        D, Q = nrm(D0 - alpha * mu), nrm(Q0 - alpha * mu)
        S = Q @ D.T; rel = S[idx, qr]; M = S.copy(); M[idx, qr] = -np.inf; margin = rel - M.max(1)
        Sb = Q @ np.where(D > 0, 1.0, -1.0).T
        Sc = S - S.mean(1, keepdims=True); Sbc = Sb - Sb.mean(1, keepdims=True)
        a = (Sc * Sbc).sum(1) / (Sc * Sc).sum(1); err = Sbc / a[:, None] - Sc
        top1 = margin > 0; lost = (Sb > Sb[idx, qr][:, None]).sum(1) > 0
        for k in np.where(top1)[0]:
            qrows.append(dict(lang=l, alpha=alpha, ratio=margin[k] / err[k].std(), lost=bool(lost[k])))
q = pd.DataFrame(qrows); q.to_csv(f"results/mechanism_queries_{MODEL}.csv", index=False)
bins = [0, 0.25, 0.5, 1, 2, np.inf]; q["bin"] = pd.cut(q.ratio, bins)
print(q.groupby(["alpha", "bin"], observed=True).lost.agg(["mean", "size"]).round(3).to_string())

# 4. Share of bits two documents have in common, and spread over documents of the query-independent offset mu'b
#    against the query-dependent part (q-mu)'b of the score, with centered codes b = sign(d - mu).
brows = []
for l in langs:
    D, Q = E[l]["docs"], E[l]["queries"]; iu = np.triu_indices(len(D), 1)
    agree = lambda B: float((((B @ B.T)[iu] / B.shape[1] + 1) / 2).mean())
    B0, Bc = np.where(D > 0, 1.0, -1.0), np.where(D - mu > 0, 1.0, -1.0)
    brows.append(dict(lang=l, bits_shared_sign=agree(B0), bits_shared_centered=agree(Bc),
                      offset_sd=float((mu @ Bc.T).std()), query_part_sd=float(((Q - mu) @ Bc.T).std(1).mean())))
pd.DataFrame(brows).to_csv(f"results/mechanism_bits_{MODEL}.csv", index=False); print(pd.DataFrame(brows).mean(numeric_only=True).round(3).to_dict())

# 5. Centering one side only, with sign codes: stored bits centered (query uncentered), query centered (bits uncentered), both.
orows = []
for l in langs:
    D, Q, qr = E[l]["docs"], E[l]["queries"], E[l]["qrels"]; Qc, Dc = nrm(Q - mu), nrm(D - mu)
    base = retrieval_metrics(Q @ D.T, qr)["ndcg_at_10"]; B0, Bc = np.where(D > 0, 1.0, -1.0), np.where(Dc > 0, 1.0, -1.0)
    loss = lambda q, b: 100 * (base - retrieval_metrics(q @ b.T, qr)["ndcg_at_10"]) / base
    orows.append(dict(lang=l, base_ndcg=base, loss_sign=loss(Q, B0), loss_bits_only=loss(Q, Bc), loss_query_only=loss(Qc, B0), loss_both=loss(Qc, Bc)))
pd.DataFrame(orows).to_csv(f"results/mechanism_onesided_{MODEL}.csv", index=False); print(pd.DataFrame(orows).mean(numeric_only=True).round(2).to_dict())
