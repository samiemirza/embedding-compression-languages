def nz(v, nd=1):
    v = round(float(v), nd)
    return 0.0 if v == 0 else v

"""Extra LaTeX tables for the TMLR version: per-language appendix, symmetric-vs-asymmetric, full calibration."""
import pandas as pd, numpy as np
MODELS = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
NICE = {"me5-small": "E5-small", "me5-base": "E5-base", "me5-large": "E5-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
df = pd.read_csv("results/all_results_merged.csv", low_memory=False)
a = df[df.qmode == "asym"]; meta = pd.read_csv("data/lang_meta.csv").set_index("lang")
esc = lambda s: str(s).replace("_", "\\_").replace("&", "\\&")

# A. per-language appendix: script, Joshi, margin (mE5-small), binary loss per model
f32 = a[a.scheme == "fp32"].pivot_table(index="lang", columns="model", values="margin_mean")
bl = a[a.scheme == "binary"].pivot_table(index="lang", columns="model", values="drop_pct")
i2 = a[(a.scheme == "int2_p5") & (a.calib == "english")].pivot_table(index="lang", columns="model", values="drop_pct")
L = ["\\begin{small}", "\\begin{longtable}{llr" + "r" * (2 * len(MODELS)) + "}",
     "\\caption{Per-language float32 margin ($\\mu$) and relative loss (\\%) under binarisation (bin) and int2-clip (int2) for every model; J = Joshi class.}\\label{tab:perlang}\\\\", "\\toprule",
     "language & script & J & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NICE[m]}}}" for m in MODELS) + " \\\\",
     " & & & " + " & ".join(["bin & int2"] * len(MODELS)) + " \\\\", "\\midrule", "\\endfirsthead", "\\toprule",
     "language & script & J & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NICE[m]}}}" for m in MODELS) + " \\\\",
     " & & & " + " & ".join(["bin & int2"] * len(MODELS)) + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
for lang in sorted(bl.index):
    j = meta.loc[lang, "joshi_class"]; j = "--" if pd.isna(j) else str(int(j))
    L.append(f"{esc(meta.loc[lang,'name'])} & {meta.loc[lang,'script']} & {j} & " + " & ".join(f"{nz(bl.loc[lang,m],0):.0f} & {nz(i2.loc[lang,m],0):.0f}" for m in MODELS) + " \\\\")
L += ["\\end{longtable}", "\\end{small}"]
open("paper_tmlr/tab_perlang.tex", "w").write("\n".join(L))

# B. symmetric vs asymmetric
rows = []
for m in MODELS:
    r = {"model": NICE[m]}
    for sc, cal, lab in [("int4", "english", "int4"), ("int2_p5", "english", "int2-clip"), ("binary", "none", "binary")]:
        for q in ["asym", "sym"]:
            s = df[(df.model == m) & (df.scheme == sc) & (df.calib == cal) & (df.qmode == q)]
            r[f"{lab} {q}"] = s.drop_pct.mean()
    rows.append(r)
t = pd.DataFrame(rows)
hdr = "model & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l}}}" for l in ["int4", "int2-clip", "binary"]) + " \\\\\n & " + " & ".join(["query fp32 & query compressed"] * 3) + " \\\\"
body = "\n".join(f"{r.model} & " + " & ".join(f"{nz(r[c]):.1f}" for c in t.columns[1:]) + " \\\\" for _, r in t.iterrows())
open("paper_tmlr/tab_sym.tex", "w").write("\\begin{table}[t]\\centering\\footnotesize\n\\begin{tabular}{lrrrrrr}\n\\toprule\n" + hdr + "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\caption{Mean relative loss (\\%) over 122 languages when only the corpus is compressed (query float32) versus when the query is compressed the same way.}\\label{tab:sym}\n\\end{table}\n")

# C. full calibration table per model, low-resource and all
CAL = [("english", "English"), ("mixed", "mixed (LOO)"), ("mixed_lat", "Latin-only"), ("target64", "target, 64"), ("target", "target, all")]
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{lll" + "r" * len(CAL) + "}", "\\toprule", "model & scheme & languages & " + " & ".join(c[1] for c in CAL) + " \\\\", "\\midrule"]
for m in MODELS:
    for sc, lab in [("int2", "int2"), ("int2_p5", "int2-clip"), ("binary_cal", "bin-centred")]:
        for g, mask in [("all", a.lang.notna()), ("low-res.", a.joshi_class <= 1)]:
            s = a[(a.model == m) & (a.scheme == sc) & mask]
            L.append((NICE[m] if (sc == "int2" and g == "all") else "") + f" & {lab if g=='all' else ''} & {g} & " + " & ".join(f"{nz(s[s.calib==c].drop_pct.mean()):.1f}" for c, _ in CAL) + " \\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"
L += ["\\end{tabular}", "\\caption{Loss (\\%) by calibration source for every model (1388 vectors except ``target, 64''; LOO = leave-target-language-out).}\\label{tab:calib_full}", "\\end{table}"]
open("paper_tmlr/tab_calib_full.tex", "w").write("\n".join(L))
print("wrote paper_tmlr/tab_perlang.tex, tab_sym.tex, tab_calib_full.tex")
