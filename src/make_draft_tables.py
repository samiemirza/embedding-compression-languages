def nz(v, nd=1):
    v = round(float(v), nd)
    return 0.0 if v == 0 else v

"""Compact LaTeX tables for the paper draft (paper/draft_tab*.tex), generated from results/*.csv."""
import os
import pandas as pd, numpy as np
from scipy.stats import spearmanr
TABDIR = os.environ.get("TABDIR", "paper_tmlr")
_PFX = os.environ.get("MODEL_LABEL_PREFIX", "E5")
MODELS = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
NICE = {"me5-small": f"{_PFX}-small", "me5-base": f"{_PFX}-base", "me5-large": f"{_PFX}-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
df = pd.read_csv("results/all_results_merged.csv"); a = df[df.qmode == "asym"]
M = [m for m in MODELS if m in set(a.model)]
SC = [("int8", "english", "int8"), ("int4", "english", "int4"), ("int2", "english", "int2"), ("int2_p5", "english", "int2-clip"), ("binary", "none", "bin-sign"), ("binary_cal", "english", "bin-centred")]

L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{ll" + "r" * len(SC) + "}", "\\toprule", "model & languages & " + " & ".join(s[2] for s in SC) + " \\\\", "\\midrule"]
for m in M:
    for g, mask in [("English", a.lang == "eng_Latn"), ("all 122", a.lang.notna()), ("low-res.", a.joshi_class <= 1)]:
        vals = [a[(a.model == m) & (a.scheme == sc) & (a.calib == cal) & mask].drop_pct.mean() for sc, cal, _ in SC]
        L.append((NICE[m] if g == "English" else "") + " & " + g + " & " + " & ".join(f"{nz(v):.1f}" for v in vals) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Relative nDCG@10 loss (\\%) w.r.t.\\ float32. Corpus compressed, query float32; integer ranges and binarisation thresholds calibrated on English. low-res.\\ = 47 languages of Joshi class 0--1.}\\label{tab:drop}", "\\end{table}"]
open(f"{TABDIR}/tab_drop.tex", "w").write("\n".join(L))

f32 = a[a.scheme == "fp32"]
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{lrrrrrrr}", "\\toprule", "model & $d$ & nDCG@10 & $\\bar{c}$ & $\\mu_{\\mathrm{eng}}$ & $\\mu_{\\mathrm{low}}$ & $\\rho_{\\mathrm{bin}}$ & $\\rho_{\\mathrm{int2}}$ \\\\", "\\midrule"]
for m in M:
    s = f32[f32.model == m]; b = a[(a.model == m) & (a.scheme == "binary")]; i = a[(a.model == m) & (a.scheme == "int2_p5") & (a.calib == "english")]
    L.append(f"{NICE[m]} & {int(s.dim.iloc[0])} & {s.base_ndcg.mean():.3f} & {s.doc_cos_mean.mean():.2f} & {s[s.lang=='eng_Latn'].margin_mean.mean():.3f} & {s[s.joshi_class<=1].margin_mean.mean():.3f} & {spearmanr(b.margin_mean,b.drop_pct).statistic:.2f} & {spearmanr(i.margin_mean,i.drop_pct).statistic:.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Embedding geometry per model. $\\bar{c}$: mean cosine between passages (mean over languages); $\\mu$: mean float32 margin for English and for low-resource languages; $\\rho$: Spearman correlation across the 122 languages between margin and loss under binarisation / int2-clip.}\\label{tab:geom}", "\\end{table}"]
open(f"{TABDIR}/tab_geometry.tex", "w").write("\n".join(L))

CAL = [("english", "English"), ("mixed", "mixed (LOO)"), ("mixed_lat", "Latin-only"), ("target64", "target, 64"), ("target", "target, all")]
e5 = a[a.model.isin(["me5-small", "me5-base", "me5-large"])]
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{ll" + "r" * len(CAL) + "}", "\\toprule", "scheme & languages & " + " & ".join(c[1] for c in CAL) + " \\\\", "\\midrule"]
for sc, lab in [("int2", "int2"), ("int2_p5", "int2-clip"), ("binary_cal", "bin-centred")]:
    for g, mask in [("all 122", e5.lang.notna()), ("low-res.", e5.joshi_class <= 1)]:
        vals = [e5[(e5.scheme == sc) & (e5.calib == c) & mask].drop_pct.mean() for c, _ in CAL]
        L.append((lab if g == "all 122" else "") + " & " + g + " & " + " & ".join(f"{nz(v):.1f}" for v in vals) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{Loss (\\%) by calibration source, {_PFX} models pooled; 1388 vectors except ``target, 64''; LOO = leave-target-language-out.}}\\label{{tab:calib}}", "\\end{table}"]
open(f"{TABDIR}/tab_calib.tex", "w").write("\n".join(L))
print("wrote tab_drop/tab_geometry/tab_calib to", TABDIR)
