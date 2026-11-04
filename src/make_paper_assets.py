"""Figure 1 and Tables 1-2 of the paper, written to paper/ (fig_margin.pdf, tab_loss.tex, tab_storage.tex), from results/*.csv only."""
import os
import pandas as pd, matplotlib
os.makedirs("paper", exist_ok=True)
matplotlib.use("Agg"); matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.linewidth": .5,
    "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2, "ytick.major.size": 2})
import matplotlib.pyplot as plt
M = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
N = {"me5-small": "E5-small", "me5-base": "E5-base", "me5-large": "E5-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
COL = dict(zip(M, ["#184f95", "#eb6834", "#1baf7a", "#6250d6", "#eda100"])); MK = dict(zip(M, ["o", "s", "^", "D", "v"]))
OPEN = {"bge-m3", "qwen3-0.6b"}  # open markers keep the two families apart in black-and-white print
INK, MUTED = "#0b0b0b", "#52514e"
meta = pd.read_csv("data/lang_meta.csv").set_index("lang"); low = set(meta[meta.joshi_class <= 1].index)
df = pd.read_csv("results/all_results_merged.csv", low_memory=False); a = df[df.qmode == "asym"]

fig, axes = plt.subplots(1, 2, figsize=(4.78, 2.05))
for ax, (sc, cal, title) in zip(axes, [("int2_p5", "english", "int2-clip (2 bits)"), ("binary", "none", "sign binarization (1 bit)")]):
    for m in M:
        s = a[(a.model == m) & (a.scheme == sc) & (a.calib == cal)]
        ax.scatter(s.margin_mean, s.drop_pct, s=14, marker=MK[m], label=N[m], zorder=2, **(dict(facecolors="none", edgecolors=COL[m], linewidths=.6) if m in OPEN else dict(c=COL[m], linewidths=0)))
    for m in M:
        e = a[(a.model == m) & (a.scheme == sc) & (a.calib == cal) & (a.lang == "eng_Latn")].iloc[0]
        ax.scatter(e.margin_mean, e.drop_pct, s=34, facecolors="none", edgecolors=INK, linewidths=.7, marker=MK[m], zorder=3)
    ax.set_title(title, fontsize=8, color=INK, pad=3); ax.set_xlabel("mean float32 margin of the language", color=MUTED)
    ax.axvline(0, color=MUTED, lw=.4, ls=":", zorder=1); ax.grid(axis="y", color="#e8e8e4", lw=.4, zorder=0); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)
axes[0].set_ylabel("loss of nDCG@10 (%)", color=MUTED)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5, handletextpad=.1, columnspacing=.9, markerscale=1.5, bbox_to_anchor=(.5, -.01))
fig.tight_layout(pad=.25, rect=(0, .1, 1, 1)); fig.savefig("paper/fig_margin.pdf")

fp = a[a.scheme == "fp32"].set_index(["model", "lang"])
L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lrrrrrrrrr}", "\\hline",
     " & & & & \\multicolumn{3}{c}{int2-clip} & \\multicolumn{3}{c}{sign} \\\\", "model & $d$ & $\\bar c$ & $\\rho$ & Eng & all & low & Eng & all & low \\\\", "\\hline"]
from scipy.stats import spearmanr
for m in M:
    f = fp.loc[m]; b = a[(a.model == m) & (a.scheme == "binary")].set_index("lang").drop_pct; c = a[(a.model == m) & (a.scheme == "int2_p5") & (a.calib == "english")].set_index("lang").drop_pct
    g = lambda s: (s["eng_Latn"], s.mean(), s[s.index.isin(low)].mean())
    d = int(f.dim.iloc[0]); rho = spearmanr(f.margin_mean, b.reindex(f.index))[0]
    L.append(f"{N[m]} & {d} & {f.doc_cos_mean.mean():.2f} & ${rho:.2f}$ & " + " & ".join(f"{v:.1f}" for v in g(c) + g(b)) + " \\\\")
L += ["\\hline", "\\end{tabular}", "\\caption{Loss (\\%) relative to float32 on Belebele for English, all 122 languages and the 47 low-resource languages. $\\bar c$: mean passage cosine; $\\rho$: Spearman correlation of margin and sign loss over languages.}\\label{tab:loss}", "\\end{table}"]
open("paper/tab_loss.tex", "w").write("\n".join(L) + "\n")

L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lrrrrrrrrr}", "\\hline",
     " & \\multicolumn{2}{c}{sign} & \\multicolumn{3}{c}{centered sign} & \\multicolumn{2}{c}{PQ} & cent.\\ PQ & PCA \\\\", "model & all & low & all & low & own & all & low & all & all \\\\", "\\hline"]
for m in M:
    x = pd.read_csv(f"results/extra_belebele_{m}.csv")
    w = lambda d: (d.loss.mean(), d[d.lang.isin(low)].loss.mean())
    s = x[x.method == "check_sign"]; c = x[(x.method == "center_sign") & (x.n == 1388)]
    p = x[(x.method == "pq") & (x.budget == "1bit") & (x.calib == "english")]; q = x[(x.method == "pq_c") & (x.budget == "1bit") & (x.calib == "english")]; r = x[(x.method == "pca8") & (x.budget == "1bit") & (x.calib == "english")]
    v = [*w(s), *w(c), c.loss_vs_t.mean(), *w(p), q.loss.mean(), r.loss.mean()]
    L.append(N[m] + " & " + " & ".join(f"{u:.1f}" for u in v) + " \\\\")
L += ["\\hline", "\\end{tabular}", "\\caption{Loss (\\%) on Belebele at 1 bit per dimension ($d/8$ bytes), relative to the original float32 quality; own: relative to the centered float32 quality. PQ, centered PQ and PCA + int8 are fitted on English; queries are not quantized.}\\label{tab:storage}", "\\end{table}"]
open("paper/tab_storage.tex", "w").write("\n".join(L) + "\n")

print("wrote")
