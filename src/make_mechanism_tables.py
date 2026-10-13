"""Appendix tables for the mechanism, held-out calibration and robustness analyses (paper_acl/tab_graded.tex,
tab_flip.tex, tab_heldout.tex, tab_dist.tex), from results/mechanism*_me5-small.csv,
results/heldout_calibration_me5-small.csv and results/all_results_merged.csv."""
import numpy as np, pandas as pd
meta = pd.read_csv("data/lang_meta.csv").set_index("lang")
low = set(meta[meta.joshi_class <= 1].index)
W = lambda p, L: open(f"paper_acl/{p}", "w").write("\n".join(L) + "\n")

m = pd.read_csv("results/mechanism_me5-small.csv")
q = pd.read_csv("results/mechanism_queries_me5-small.csv")
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{rrrrrrr}", "\\toprule",
     "$\\alpha$ & $\\bar{c}$ & margin & perturb. & float32 & loss & loss own \\\\", "\\midrule"]
for a, d in m.groupby("alpha"):
    L.append(f"{a:.2f} & {d.doc_cos.mean():.2f} & {d.margin.mean():.3f} & {d.noise.mean():.3f} & {d.ndcg_fp32.mean():.3f} & {d.loss_vs_orig.mean():.1f} & {d.loss_vs_t.mean():.1f} \\\\")
L += ["\\bottomrule", "\\end{tabular}}",
      "\\caption{Graded centering, E5-small, mean over 122 languages: documents and queries are shifted by $-\\alpha\\vmu$ ($\\vmu$: English calibration mean) and re-normalized. perturb.: standard deviation over documents of the change in score caused by sign binarization (compressed scores rescaled per query to the float32 scale); float32: nDCG@10 after the shift; loss: binary loss (\\%) relative to the original float32 quality; loss own: relative to the float32 quality after the shift.}\\label{tab:graded}", "\\end{table}"]
W("tab_graded.tex", L)

bins = [0, 0.25, 0.5, 1, 2, np.inf]; labs = ["$<0.25$", "0.25--0.5", "0.5--1", "1--2", "$>2$"]
q["bin"] = pd.cut(q.ratio, bins, labels=labs)
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{lrrrr}", "\\toprule",
     " & \\multicolumn{2}{c}{not centered} & \\multicolumn{2}{c}{centered} \\\\", "margin / perturb. & lost & queries & lost & queries \\\\", "\\midrule"]
for b in labs:
    x = q[(q.alpha == 0) & (q.bin == b)]; y = q[(q.alpha == 1) & (q.bin == b)]
    L.append(f"{b} & {100 * x.lost.mean():.0f}\\% & {len(x):,} & {100 * y.lost.mean():.0f}\\% & {len(y):,} \\\\".replace(",", "{,}"))
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Share of queries ranked first at float32 that lose the first rank under sign binarization, by the ratio of the query's float32 margin to the perturbation of its scores (E5-small, all 122 languages pooled).}\\label{tab:flip}", "\\end{table}"]
W("tab_flip.tex", L)

h = pd.read_csv("results/heldout_calibration_me5-small.csv")
def grp(d): return pd.Series({"eng": d[d.lang == "eng_Latn"].loss.mean(), "all": d.loss.mean(), "low": d[d.lang.isin(low)].loss.mean()})
ps = h.groupby(["scheme", "calib", "seed"]).apply(grp, include_groups=False)
agg = ps.groupby(level=["scheme", "calib"]).agg(["mean", "std"])
SN = {"int2": "int2", "int2_clip": "int2-clip", "bin_mean_thr": "bin-mean-thr", "centered_sign": "centering + sign"}
CN = {"bele_en": "Belebele English", "wiki_en": "Wikipedia English", "mixed": "Belebele mixed"}
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llrrr}", "\\toprule", "format & calibration & English & all & low \\\\", "\\midrule"]
for i, sc in enumerate(["int2", "int2_clip", "bin_mean_thr", "centered_sign"]):
    for j, c in enumerate(["bele_en", "wiki_en", "mixed"]):
        r = agg.loc[(sc, c)]
        cell = lambda k: f"{r[(k, 'mean')]:.1f}" + ("" if np.isnan(r[(k, 'std')]) else f" $\\pm$ {r[(k, 'std')]:.1f}")
        L.append(f"{SN[sc] if j == 0 else ''} & {CN[c]} & {cell('eng')} & {cell('all')} & {cell('low')} \\\\")
    if i < 3: L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}}",
      "\\caption{Calibration without content overlap, E5-small: loss (\\%) with the Belebele English evaluation data as calibration set (one set), with 1388 English Wikipedia passages that share no content with Belebele (five draws), and with the mixed leave-target-out Belebele sample (five draws); mean $\\pm$ standard deviation over draws.}\\label{tab:heldout}", "\\end{table}"]
W("tab_heldout.tex", L)

df = pd.read_csv("results/all_results_merged.csv", low_memory=False); a = df[df.qmode == "asym"]
lw = set(meta.wiki_articles.sort_values().index[:47])
N = {"me5-small": "E5-small", "me5-base": "E5-base", "me5-large": "E5-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
L = ["\\begin{table*}[t]\\centering\\footnotesize", "\\begin{tabular}{lrrrrlrrr}", "\\toprule",
     " & \\multicolumn{5}{c}{all 122 languages} & \\multicolumn{3}{c}{low-resource gap} \\\\", "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}", "model & mean & median & 90th pct. & worst & worst language & class 0--1 & fewest articles & rest \\\\", "\\midrule"]
for k, n in N.items():
    s = a[(a.model == k) & (a.scheme == "binary")].set_index("lang").drop_pct
    L.append(f"{n} & {s.mean():.1f} & {s.median():.1f} & {s.quantile(.9):.1f} & {s.max():.1f} & {meta.loc[s.idxmax(), 'name'].replace('romanised', 'romanized')} & {s[s.index.isin(low)].mean():.1f} & {s[s.index.isin(lw)].mean():.1f} & {s[~s.index.isin(lw)].mean():.1f} \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Distribution of the binary loss (\\%) over the 122 languages, and the low-resource gap under two definitions: resource classes 0--1 \\citep{joshi2020state}, and the 47 languages with the fewest Wikipedia articles (rest: the other languages with a Wikipedia count).}\\label{tab:dist}", "\\end{table*}"]
W("tab_dist.tex", L); print("wrote tab_graded, tab_flip, tab_heldout, tab_dist")
