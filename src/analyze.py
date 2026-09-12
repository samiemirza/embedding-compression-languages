"""Analysis for the ESANN paper (margin framing).

Inputs : results/<model>.csv, embeddings/<model>/fertility.csv, data/lang_meta.csv
Outputs: results/summary_*.csv, paper/tab*.tex, paper/fig*.pdf/png
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.environ.get("FIGDIR", "paper_tmlr/figs")
os.makedirs(FIGDIR, exist_ok=True)
MODELS = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
_PFX = os.environ.get("MODEL_LABEL_PREFIX", "E5")   # "mE5" for the ESANN version
NICE = {"me5-small": f"{_PFX}-small", "me5-base": f"{_PFX}-base", "me5-large": f"{_PFX}-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
meta = pd.read_csv("data/lang_meta.csv")

frames = []
for m in MODELS:
    p = f"results/{m}.csv"
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    fert = pd.read_csv(f"embeddings/{m}/fertility.csv")
    fert["fertility"] = fert.doc_tokens / fert.set_index("lang").loc["eng_Latn"].doc_tokens
    df = df.merge(fert[["lang", "fertility"]], on="lang").merge(meta, on=["lang", "script"])
    df["dim"] = np.load(f"embeddings/{m}/eng_Latn.npz")["docs"].shape[1]
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
df["drop_pct"] = 100 * df.rel_drop
df["abs_drop"] = df.base_ndcg - df.ndcg_at_10
df["lowres"] = df.joshi_class <= 1
models = [m for m in MODELS if m in set(df.model)]
a = df[df.qmode == "asym"].copy()
df.to_csv("results/all_results_merged.csv", index=False)


def tex(df_, path, caption, label, fmt="%.1f", index=False, colfmt=None):
    s = df_.to_latex(index=index, float_format=lambda x: fmt % x, caption=caption, label=label, escape=False,
                     column_format=colfmt)
    open(path, "w").write(s.replace("_", r"\_"))


def grp(s):
    return pd.Series({"Eng": s[s.lang == "eng_Latn"].drop_pct.mean(), "All": s.drop_pct.mean(),
                      "Low": s[s.lowres].drop_pct.mean()})


# ---------------- Table 1: per-model loss, English / all / low-resource ----------------
KEY = [("fp8_e4m3", "none"), ("int8", "english"), ("int4", "english"), ("int2", "english"),
       ("int2_p5", "english"), ("binary", "none"), ("binary_cal", "english")]
LABEL = {"fp8_e4m3": "float8", "int8": "int8", "int4": "int4", "int2": "int2 (min/max)",
         "int2_p5": "int2 (pct-clip)", "binary": "binary (sign)", "binary_cal": "binary (centred)"}
rows = []
for sc, cal in KEY:
    r = {"scheme": LABEL[sc], "bits": {"fp8_e4m3": 8, "int8": 8, "int4": 4, "int2": 2, "int2_p5": 2, "binary": 1, "binary_cal": 1}[sc]}
    for m in models:
        g = grp(a[(a.model == m) & (a.scheme == sc) & (a.calib == cal)])
        for k in ["Eng", "All", "Low"]:
            r[f"{NICE[m]} {k}"] = g[k]
    rows.append(r)
t1 = pd.DataFrame(rows)
t1.to_csv("results/summary_table1_per_model.csv", index=False)
print("\n=== Table 1: relative nDCG@10 drop (%) — English / all 122 / low-resource (Joshi 0-1); corpus compressed, query fp32; English calibration ===")
print(t1.round(1).to_string(index=False))
# compact latex: multicolumn header built by hand
hdr = " & ".join([f"\\multicolumn{{3}}{{c}}{{{NICE[m]}}}" for m in models])
sub = " & ".join(["Eng & All & Low"] * len(models))
body = "\n".join(" & ".join([r["scheme"], str(r["bits"])] + [f"{r[f'{NICE[m]} {k}']:.1f}" for m in models for k in ["Eng", "All", "Low"]]) + r" \\" for _, r in t1.iterrows())
open("paper/tab1_drop.tex", "w").write(
    "\\begin{table}[t]\\centering\\footnotesize\n\\begin{tabular}{l c " + "ccc " * len(models) + "}\n\\toprule\n"
    "scheme & bits & " + hdr + " \\\\\n & & " + sub + " \\\\\n\\midrule\n" + body +
    "\n\\bottomrule\n\\end{tabular}\n\\caption{Relative nDCG@10 loss (\\%) vs.\\ float32 for English, the mean over 122 languages, and the mean over low-resource languages (Joshi class 0--1). Corpus compressed, query float32; integer ranges calibrated on English.}\\label{tab:drop}\n\\end{table}\n")

# ---------------- Table 2: model geometry explains the model gap ----------------
f32 = a[a.scheme == "fp32"]
rows = []
for m in models:
    s = f32[f32.model == m]
    b = a[(a.model == m) & (a.scheme == "binary")]
    rows.append({"model": NICE[m], "dim": int(s.dim.iloc[0]), "nDCG@10": s.base_ndcg.mean(),
                 "mean cos(doc,doc)": s.doc_cos_mean.mean(),
                 "margin Eng": s[s.lang == "eng_Latn"].margin_mean.mean(), "margin Low": s[s.lowres].margin_mean.mean(),
                 "binary loss Eng": b[b.lang == "eng_Latn"].drop_pct.mean(), "binary loss Low": b[b.lowres].drop_pct.mean(),
                 "rho(margin,loss)": spearmanr(b.margin_mean, b.drop_pct).statistic})
t2 = pd.DataFrame(rows); t2.to_csv("results/summary_table2_geometry.csv", index=False)
print("\n=== Table 2: embedding geometry per model ===")
print(t2.round(3).to_string(index=False))
tex(t2, "paper/tab2_geometry.tex", "Embedding geometry and binarisation loss per model. Margin = float32 score of the relevant passage minus the best distractor (mean over queries). $\\rho$ = Spearman correlation between per-language margin and per-language binary loss.", "tab:geom", fmt="%.3f")

# ---------------- Table 3: what predicts the per-language loss (regression) ----------------
rows = []
for sc, cal in [("int4", "english"), ("int2_p5", "english"), ("binary", "none"), ("binary_cal", "english")]:
    s = a[(a.scheme == sc) & (a.calib == cal)].dropna(subset=["joshi_class"])
    f1 = smf.ols("drop_pct ~ fertility + latin + joshi_class + C(model)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.lang})
    f2 = smf.ols("drop_pct ~ margin_mean + C(model)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.lang})
    f3 = smf.ols("drop_pct ~ margin_mean + fertility + latin + joshi_class + C(model)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.lang})
    rows.append({"scheme": LABEL[sc], "R2 covariates": f1.rsquared, "R2 margin": f2.rsquared, "R2 both": f3.rsquared,
                 "b fertility": f1.params["fertility"], "p fert": f1.pvalues["fertility"],
                 "b Latin": f1.params["latin"], "p Latin": f1.pvalues["latin"],
                 "b Joshi": f1.params["joshi_class"], "p Joshi": f1.pvalues["joshi_class"],
                 "b fert|margin": f3.params["fertility"], "p fert|margin": f3.pvalues["fertility"],
                 "b Joshi|margin": f3.params["joshi_class"], "p Joshi|margin": f3.pvalues["joshi_class"]})
t3 = pd.DataFrame(rows); t3.to_csv("results/summary_table3_regression.csv", index=False)
print("\n=== Table 3: OLS of per-language loss (model fixed effects, SE clustered by language) ===")
print(t3.round(3).to_string(index=False))
tex(t3[["scheme", "R2 covariates", "R2 margin", "R2 both", "b fertility", "p fert", "b Latin", "p Latin", "b Joshi", "p Joshi"]],
    "paper/tab3_regression.tex", "Predicting per-language loss (\\%). $R^2$ with language covariates (tokenizer fertility, Latin script, Joshi class), with the float32 margin alone, and with both; all models pooled with model fixed effects.", "tab:reg", fmt="%.2f")

# ---------------- Table 4: calibration source (negative result), e5 models, low-resource ----------------
CAL = ["english", "mixed", "mixed_lat", "target64", "target"]
rows = []
for m in models:
    for sc in ["int4", "int2", "int2_p5", "binary_cal"]:
        s = a[(a.model == m) & (a.scheme == sc) & a.lowres]
        r = {"model": NICE[m], "scheme": LABEL[sc]}
        for c in CAL:
            r[c] = s[s.calib == c].drop_pct.mean()
        rows.append(r)
t4 = pd.DataFrame(rows); t4.to_csv("results/summary_table4_calibration.csv", index=False)
print("\n=== Table 4: calibration source, mean loss (%) on low-resource languages ===")
print(t4.round(1).to_string(index=False))
tex(t4[t4.model.isin(["mE5-small", "mE5-base"])], "paper/tab4_calibration.tex",
    "Mean loss (\\%) on low-resource languages by calibration source (1388 vectors; target64 = 64 target-language texts; target = in-language oracle).", "tab:calib")

# ---------------- Table 5: same language, two scripts ----------------
rows = []
for r in meta[meta.romanised == 1].lang:
    base = r.split("_")[0]
    for n in [l for l in meta.lang if l.startswith(base + "_") and l != r]:
        if n not in set(a.lang) or r not in set(a.lang):
            continue
        row = {"language": meta.set_index("lang").loc[n, "name"], "native": n.split("_")[1]}
        for sc, cal in [("int2_p5", "english"), ("binary", "none")]:
            e5 = a[a.model.isin(["me5-small", "me5-base"]) & (a.scheme == sc) & (a.calib == cal)].set_index("lang").drop_pct
            row[f"{LABEL[sc]} native"] = e5.loc[[n]].mean(); row[f"{LABEL[sc]} romanised"] = e5.loc[[r]].mean()
        m0 = f32[f32.model.isin(["me5-small", "me5-base"])].set_index("lang").margin_mean
        row["margin native"] = m0.loc[[n]].mean(); row["margin romanised"] = m0.loc[[r]].mean()
        rows.append(row)
t5 = pd.DataFrame(rows); t5.to_csv("results/summary_table5_script_pairs.csv", index=False)
print("\n=== Table 5: native vs romanised script (mE5 models pooled) ===")
print(t5.round(3).to_string(index=False))
tex(t5, "paper/tab5_script_pairs.tex", "Same language, two scripts (mE5 models pooled): loss (\\%) and float32 margin.", "tab:script", fmt="%.2f")

# ---------------- Figure 1 (main): loss vs margin, per language, coloured by model ----------------
COL = {"me5-small": "tab:red", "me5-base": "tab:orange", "me5-large": "tab:brown", "bge-m3": "tab:blue", "qwen3-0.6b": "tab:green"}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, (sc, cal) in zip(axes, [("int4", "english"), ("int2_p5", "english"), ("binary", "none")]):
    for m in models:
        s = a[(a.model == m) & (a.scheme == sc) & (a.calib == cal)]
        ax.scatter(s.margin_mean, s.drop_pct, s=10, c=COL[m], alpha=.65, label=NICE[m])
        e = s[s.lang == "eng_Latn"].iloc[0]; ax.scatter(e.margin_mean, e.drop_pct, s=60, facecolors="none", edgecolors=COL[m], linewidths=1.2)
    ax.set_title(LABEL[sc]); ax.set_xlabel("float32 score margin"); ax.axvline(0, color="k", lw=.5, ls=":")
axes[0].set_ylabel("relative nDCG@10 loss (%)"); axes[0].legend(fontsize=7, title="circle = English", title_fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig1_loss_vs_margin.pdf")

# ---------------- Figure 2: loss by bits, English vs low-resource, per model ----------------
fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 3), sharey=True)
order = [("int8", "english"), ("int4", "english"), ("int2_p5", "english"), ("binary_cal", "english"), ("binary", "none")]
for ax, m in zip(np.atleast_1d(axes), models):
    for lab, mask, col in [("English", a.lang == "eng_Latn", "tab:blue"), ("all 122", a.lang.notna(), "tab:gray"), ("low-resource", a.lowres, "tab:red")]:
        ys = [a[(a.model == m) & (a.scheme == sc) & (a.calib == cal) & mask].drop_pct.mean() for sc, cal in order]
        ax.plot(range(len(order)), ys, "o-", color=col, label=lab, ms=4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["int8", "int4", "int2\npct", "bin\ncentred", "bin\nsign"], fontsize=7)
    ax.set_title(NICE[m]); ax.set_yscale("symlog", linthresh=1); ax.grid(alpha=.3)
np.atleast_1d(axes)[0].set_ylabel("relative loss (%) [symlog]"); np.atleast_1d(axes)[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig2_loss_by_bits.pdf")

# ---------------- Figure 3: language covariates (secondary) ----------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
s = a[(a.scheme == "binary")].groupby(["lang", "model"]).drop_pct.mean().unstack("model")
mm = meta.set_index("lang")
for ax, m in zip(axes, ["me5-small", "bge-m3"]):
    if m not in s: continue
    x = s[m]; fert = a[(a.model == m) & (a.scheme == "fp32")].set_index("lang").fertility.reindex(x.index)
    for lat, col, lab in [(1, "tab:blue", "Latin"), (0, "tab:red", "non-Latin")]:
        idx = mm.loc[x.index].latin == lat
        ax.scatter(fert[idx], x[idx], s=10, c=col, alpha=.7, label=lab)
    rom = mm.loc[x.index].romanised == 1
    ax.scatter(fert[rom], x[rom], s=30, marker="x", c="k", label="romanised")
    ax.set_title(f"{NICE[m]}, binary"); ax.set_xlabel("tokens / English tokens")
axes[0].set_ylabel("relative loss (%)"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig3_loss_vs_fertility.pdf")
for old in ["fig1_drop_vs_fertility", "fig2_calibration_by_resource", "fig3_drop_vs_margin"]:
    for ext in ["pdf", "png"]:
        p = f"paper/{old}.{ext}"
        if os.path.exists(p): os.remove(p)
print("\nwrote paper/tab1..tab5 and fig1..fig3")
