"""Figures 1-3 of the NAACL version from results/all_results_merged.csv (src/analyze.py needs the raw embeddings).

Writes paper_acl/figs/fig1_loss_vs_margin.pdf, fig2_loss_by_bits.pdf, fig3_loss_vs_fertility.pdf with embedded TrueType fonts.
"""
import pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
FIGDIR = "paper_acl/figs"
MODELS = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
NICE = {"me5-small": "E5-small", "me5-base": "E5-base", "me5-large": "E5-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
LABEL = {"int4": "int4", "int2_p5": "int2-clip", "binary": "binary (sign)"}
COL = {"me5-small": "tab:red", "me5-base": "tab:orange", "me5-large": "tab:brown", "bge-m3": "tab:blue", "qwen3-0.6b": "tab:green"}
df = pd.read_csv("results/all_results_merged.csv", low_memory=False); a = df[df.qmode == "asym"].copy()
meta = pd.read_csv("data/lang_meta.csv")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, (sc, cal) in zip(axes, [("int4", "english"), ("int2_p5", "english"), ("binary", "none")]):
    for m in MODELS:
        s = a[(a.model == m) & (a.scheme == sc) & (a.calib == cal)]
        ax.scatter(s.margin_mean, s.drop_pct, s=10, c=COL[m], alpha=.65, label=NICE[m])
        e = s[s.lang == "eng_Latn"].iloc[0]
        ax.scatter(e.margin_mean, e.drop_pct, s=60, facecolors="none", edgecolors=COL[m], linewidths=1.2)
    ax.set_title(LABEL[sc]); ax.set_xlabel("float32 score margin"); ax.axvline(0, color="k", lw=.5, ls=":")
axes[0].set_ylabel("relative nDCG@10 loss (%)"); axes[0].legend(fontsize=7, title="circle = English", title_fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig1_loss_vs_margin.pdf")

fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
order = [("int8", "english"), ("int4", "english"), ("int2_p5", "english"), ("binary_cal", "english"), ("binary", "none")]
for ax, m in zip(axes, MODELS):
    for lab, mask, col in [("English", a.lang == "eng_Latn", "tab:blue"), ("all 122", a.lang.notna(), "tab:gray"), ("low-resource", a.lowres, "tab:red")]:
        ys = [a[(a.model == m) & (a.scheme == sc) & (a.calib == cal) & mask].drop_pct.mean() for sc, cal in order]
        ax.plot(range(5), ys, "o-", color=col, label=lab, ms=4)
    ax.set_xticks(range(5)); ax.set_xticklabels(["int8", "int4", "int2\nclip", "bin\nmean-thr", "bin\nsign"], fontsize=7)
    ax.set_title(NICE[m]); ax.set_yscale("symlog", linthresh=1); ax.grid(alpha=.3)
axes[0].set_ylabel("relative loss (%) [symlog]"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig2_loss_by_bits.pdf")

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
s = a[a.scheme == "binary"].groupby(["lang", "model"]).drop_pct.mean().unstack("model"); mm = meta.set_index("lang")
for ax, m in zip(axes, ["me5-small", "bge-m3"]):
    x = s[m]; fert = a[(a.model == m) & (a.scheme == "fp32")].set_index("lang").fertility.reindex(x.index)
    for lat, col, lab in [(1, "tab:blue", "Latin"), (0, "tab:red", "non-Latin")]:
        idx = mm.loc[x.index].latin == lat; ax.scatter(fert[idx], x[idx], s=10, c=col, alpha=.7, label=lab)
    rom = mm.loc[x.index].romanised == 1; ax.scatter(fert[rom], x[rom], s=30, marker="x", c="k", label="romanized")
    ax.set_title(f"{NICE[m]}, binary"); ax.set_xlabel("tokens / English tokens")
axes[0].set_ylabel("relative loss (%)"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig3_loss_vs_fertility.pdf"); print("wrote fig1-3")
