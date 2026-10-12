def nz(v, nd=1):
    v = round(float(v), nd)
    return 0.0 if v == 0 else v

"""Tables and figure for the centering/whitening and rescoring experiments (written to $TABDIR and $FIGDIR, default build/)."""
import os, pandas as pd, numpy as np
os.makedirs(os.environ.get("TABDIR", "build"), exist_ok=True); os.makedirs(os.environ.get("FIGDIR", "build/figs"), exist_ok=True)
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42; import matplotlib.pyplot as plt
M = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
N = {"me5-small": "E5-small", "me5-base": "E5-base", "me5-large": "E5-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
meta = pd.read_csv("data/lang_meta.csv")[["lang", "joshi_class"]]
w = pd.read_csv("results/whitening.csv").merge(meta, on="lang"); w["low"] = w.joshi_class <= 1
r = pd.read_csv("results/rescoring.csv").merge(meta, on="lang"); r["low"] = r.joshi_class <= 1

# ---- whitening table: per model, transform x calib: fp32 nDCG, doc cos, margin, binary loss (vs orig / vs transformed), int2-clip loss vs orig, low-res binary
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{llrrrrrrr}", "\\toprule",
     "model & transform (calib.) & nDCG$_{32}$ & $\\bar{c}$ & margin & bin all & bin low & bin vs.\\ $\\mathrm{nDCG}_{32}'$ & int2-clip all \\\\", "\\midrule"]
for m in M:
    for tr, cal, lab in [("none", "english", "none"), ("center", "english", "centre (Eng)"), ("center", "mixed", "centre (mixed)"), ("zca", "english", "ZCA (Eng)"), ("zca", "mixed", "ZCA (mixed)")]:
        s = w[(w.model == m) & (w["transform"] == tr) & (w.calib == cal)]
        L.append((N[m] if tr == "none" else "") + f" & {lab} & {s.base_ndcg_t.mean():.3f} & {s.doc_cos_t.mean():.2f} & {s.margin_t.mean():.3f} & {nz(s.binary_loss_vs_orig.mean()):.1f} & {nz(s[s.low].binary_loss_vs_orig.mean()):.1f} & {nz(s.binary_loss_vs_t.mean()):.1f} & {nz(s.int2_p5_loss_vs_orig.mean()):.1f} \\\\")
    if m != M[-1]: L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Centring and whitening before compression. nDCG$_{32}$: float32 quality after the transform (mean over 122 languages); $\\bar{c}$: mean passage cosine; margin: mean float32 margin. Losses are relative to the \\emph{original} float32 quality, except the column marked vs.\\ nDCG$_{32}'$, which is relative to the transformed float32 quality. Negative values are gains over the original float32 baseline.}\\label{tab:whiten}", "\\end{table}"]
open(os.environ.get("TABDIR","build")+"/tab_whiten.tex", "w").write("\n".join(L))

# ---- rescoring table: binary / int2-clip, int8 rescoring, k in 10,20,50,100; English / all / low per model
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{ll" + "rrr" * 4 + "}", "\\toprule",
     "model & index & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$k={k}$}}" for k in [10, 20, 50, 100]) + " \\\\",
     " & & " + " & ".join(["Eng & all & low"] * 4) + " \\\\", "\\midrule"]
for m in M:
    for sc, lab in [("binary", "binary"), ("int2_p5", "int2-clip")]:
        s = r[(r.model == m) & (r.scheme == sc) & (r.rescore == "int8")]
        cells = []
        for k in [10, 20, 50, 100]:
            x = s[s.k == k]; cells += [f"{nz(x[x.lang=='eng_Latn'].loss.mean()):.1f}", f"{nz(x.loss.mean()):.1f}", f"{nz(x[x.low].loss.mean()):.1f}"]
        L.append((N[m] if sc == "binary" else "") + f" & {lab} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Loss (\\%) after retrieving $k$ candidates with the compressed index and rescoring them with the float32 query against an int8 corpus. Rescoring with the float32 corpus differs by at most 0.2 points. English calibration.}\\label{tab:rescore}", "\\end{table}"]
open(os.environ.get("TABDIR","build")+"/tab_rescore.tex", "w").write("\n".join(L))

# ---- figure: left = binary loss before/after centring per model (all, low); right = rescoring loss vs k (binary, low-res) per model
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
x = np.arange(len(M)); wdt = 0.2
for i, (tr, cal, lab, col) in enumerate([("none", "english", "sign, no transform", "tab:red"), ("center", "english", "sign after centering (Eng)", "tab:orange"), ("center", "mixed", "sign after centering (mixed)", "tab:green")]):
    vals = [w[(w.model == m) & (w["transform"] == tr) & (w.calib == cal)].binary_loss_vs_orig.mean() for m in M]
    lows = [w[(w.model == m) & (w["transform"] == tr) & (w.calib == cal) & w.low].binary_loss_vs_orig.mean() for m in M]
    ax[0].bar(x + (i - 1) * wdt, vals, wdt, color=col, label=lab)
    ax[0].scatter(x + (i - 1) * wdt, lows, marker="_", color="k", s=80, zorder=3)
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xticks(x); ax[0].set_xticklabels([N[m] for m in M], fontsize=8); ax[0].set_ylabel("binary loss vs. original float32 (%)")
ax[0].legend(fontsize=7); ax[0].set_title("centering before binarization (bar: all 122; tick: low-resource)", fontsize=9)
COL = {"me5-small": "tab:red", "me5-base": "tab:orange", "me5-large": "tab:brown", "bge-m3": "tab:blue", "qwen3-0.6b": "tab:green"}
for m in M:
    s = r[(r.model == m) & (r.scheme == "binary") & (r.rescore == "int8") & r.low].groupby("k").loss.mean()
    ax[1].plot(s.index, s.values, "o-", color=COL[m], label=N[m], ms=4)
ax[1].set_xscale("log"); ax[1].set_xticks([10, 20, 50, 100]); ax[1].set_xticklabels([10, 20, 50, 100]); ax[1].set_xlabel("candidates k rescored with int8"); ax[1].set_ylabel("binary loss, low-resource (%)")
ax[1].legend(fontsize=7); ax[1].set_title("rescoring", fontsize=9); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.environ.get("FIGDIR","build/figs")+"/fig4_centring_rescoring.pdf")
print("wrote tab_whiten, tab_rescore, fig4")
