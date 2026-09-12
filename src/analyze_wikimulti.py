def nz(v, nd=1):
    v = round(float(v), nd)
    return 0.0 if v == 0 else v

"""Second-task analysis: WikipediaRetrievalMultilingual (16 languages, ~13.5k passages each).
Compares each model's loss and the margin relation with Belebele on the same 16 languages.
Outputs results/summary_wikimulti.csv and paper/draft_tab5_wikimulti.tex"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import sys; sys.path.insert(0, os.path.dirname(__file__))
from data import WIKI_CODE

MODELS = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
_PFX = os.environ.get("MODEL_LABEL_PREFIX", "E5")
TABDIR = os.environ.get("TABDIR", "paper_tmlr")
NICE = {"me5-small": f"{_PFX}-small", "me5-base": f"{_PFX}-base", "me5-large": f"{_PFX}-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
meta = pd.read_csv("data/lang_meta.csv")
codes = list(WIKI_CODE.values())
SC = [("int4", "english", "int4"), ("int2_p5", "english", "int2-clip"), ("binary", "none", "bin-sign"), ("binary_cal", "english", "bin-centred")]

def load(task):
    fs = [f"results/{('' if task=='belebele' else task+'_')}{m}.csv" for m in MODELS]
    fs = [f for f in fs if os.path.exists(f)]
    df = pd.concat([pd.read_csv(f) for f in fs]); df["drop_pct"] = 100 * df.rel_drop
    df = df[df.qmode == "asym"].merge(meta[["lang", "joshi_class"]], on="lang")
    return df[df.lang.isin(codes)]

w, b = load("wikimulti"), load("belebele")
rows = []
for m in [m for m in MODELS if m in set(w.model)]:
    for task, df in [("Wikipedia", w), ("Belebele", b)]:
        s = df[df.model == m]
        f32 = s[s.scheme == "fp32"]
        r = {"model": NICE[m], "task": task, "n_langs": s.lang.nunique(), "nDCG@10": f32.base_ndcg.mean(), "margin": f32.margin_mean.mean()}
        for sc, cal, lab in SC:
            x = s[(s.scheme == sc) & (s.calib == cal)]
            r[f"{lab} Eng"] = x[x.lang == "eng_Latn"].drop_pct.mean(); r[f"{lab} all"] = x.drop_pct.mean()
            r[f"{lab} rho"] = spearmanr(x.margin_mean, x.drop_pct).statistic if len(x) > 3 else np.nan
        rows.append(r)
t = pd.DataFrame(rows); t.to_csv("results/summary_wikimulti.csv", index=False)
pd.set_option("display.width", 250); print(t.round(2).to_string(index=False))
# per-language agreement between tasks (binary loss)
wb = w[(w.scheme == "binary")].groupby(["model", "lang"]).drop_pct.mean()
bb = b[(b.scheme == "binary")].groupby(["model", "lang"]).drop_pct.mean()
j = pd.concat([wb.rename("wiki"), bb.rename("bele")], axis=1).dropna()
for m, g in j.groupby(level=0):
    print(f"{NICE[m]}: Spearman(binary loss Wikipedia vs Belebele over {len(g)} langs) = {spearmanr(g.wiki, g.bele).statistic:.2f}")
# compact LaTeX
L = ["\\begin{table}[t]\\centering\\footnotesize", "\\begin{tabular}{llrrrrrrr}", "\\toprule",
     "model & task & nDCG & margin & int2-clip & bin-sign & bin-centred & $\\rho_{\\mathrm{bin}}$ \\\\", "\\midrule"]
for _, r in t.iterrows():
    L.append(f"{r.model if r.task=='Wikipedia' else ''} & {r.task} & {r['nDCG@10']:.3f} & {r.margin:.3f} & {nz(r['int2-clip all']):.1f} & {nz(r['bin-sign all']):.1f} & {nz(r['bin-centred all']):.1f} & {r['bin-sign rho']:.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\caption{Wikipedia (13.5k passages per language) vs.\\ Belebele, same 16 languages: mean loss (\\%); $\\rho$ = Spearman(margin, binary loss).}\\label{tab:wiki}", "\\end{table}"]
open(f"{TABDIR}/tab_wiki.tex", "w").write("\n".join(L)); print("wrote", f"{TABDIR}/tab_wiki.tex")
