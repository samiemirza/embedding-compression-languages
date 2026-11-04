"""Recomputes the numbers in the paper's prose, section by section, from results/*.csv and data/lang_meta.csv only.
Figure 1 and Tables 1-2 come from make_paper_assets.py. Run from the repository root; takes about a minute."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr

M = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]; E5 = M[:3]; OTHER = M[3:]
meta = pd.read_csv("data/lang_meta.csv").set_index("lang")
low = set(meta[meta.joshi_class <= 1].index)
df = pd.read_csv("results/all_results_merged.csv", low_memory=False); a = df[df.qmode == "asym"]; sym = df[df.qmode == "sym"]
X = {m: pd.read_csv(f"results/extra_belebele_{m}.csv") for m in M}
XW = {m: pd.read_csv(f"results/extra_wikimulti_{m}.csv") for m in E5}
W = pd.read_csv("results/whitening.csv"); R = pd.read_csv("results/rescoring.csv")
rng = np.random.default_rng(0)


def show(label, value):
    if isinstance(value, dict):
        value = {k: round(float(v), 3) for k, v in value.items()}
    elif isinstance(value, (float, np.floating)):
        value = round(float(value), 3)
    print(f"  {label}: {value}")


def loss(m, scheme, calib=None, frame=a):
    x = frame[(frame.model == m) & (frame.scheme == scheme)]
    return (x if calib is None else x[x.calib == calib]).set_index("lang").drop_pct


def lo(s): return s[s.index.isin(low)]
def rest(s): return s[~s.index.isin(low)]
def per_model(f, models=M): return {m: f(m) for m in models}
def sign(m): return loss(m, "binary")
def centered(m, x=X): return x[m][(x[m].method == "center_sign") & (x[m].n == 1388)].set_index("lang")
def plain(m, x=X): return x[m][x[m].method == "check_sign"].set_index("lang")
def rows(m, method, calib="english"): return X[m][(X[m].method == method) & (X[m].budget == "1bit") & (X[m].calib == calib)].set_index("lang")


print("Method and setup")
show("languages, low-resource, others", (len(sign(M[0])), len(lo(sign(M[0]))), len(rest(sign(M[0])))))
show("standard error of the mean over languages, sign, max", max(sign(m).sem() for m in M))
show("standard error of the mean over languages, int2-clip, max", max(loss(m, "int2_p5", "english").sem() for m in M))

print("\n3.1 The loss is uneven")
show("float8 and int8 mean loss, max over models", max(max(loss(m, "fp8_e4m3").mean(), loss(m, "int8", "english").mean()) for m in M))
show("int4 mean loss, max over models", max(loss(m, "int4", "english").mean() for m in M))
show("int4 worst single language", max(loss(m, "int4", "english").max() for m in M))
show("share of other languages losing more than English, sign (%)", per_model(lambda m: 100 * (sign(m).drop("eng_Latn") > sign(m)["eng_Latn"]).mean()))
show("low-resource mean, sign", per_model(lambda m: lo(sign(m)).mean()))
show("other languages mean, sign", per_model(lambda m: rest(sign(m)).mean()))


def perm_p(s, n=20000):
    v, k = s.values, s.index.isin(low).sum(); obs = lo(s).mean() - rest(s).mean()
    hits = sum(p[:k].mean() - p[k:].mean() >= obs for p in (rng.permutation(v) for _ in range(n)))
    return (hits + 1) / (n + 1)


show("one-sided permutation p, low vs rest, sign", per_model(lambda m: perm_p(sign(m))))
few = set(meta.wiki_articles.dropna().sort_values().index[:47])
show("gap low vs rest, sign", per_model(lambda m: lo(sign(m)).mean() - rest(sign(m)).mean()))
show("gap 47 fewest Wikipedia articles vs rest, sign", per_model(lambda m: sign(m)[sign(m).index.isin(few)].mean() - sign(m)[~sign(m).index.isin(few)].mean()))
show("E5-large mean sign loss / BGE-M3, Qwen3", {m: sign("me5-large").mean() / sign(m).mean() for m in OTHER})
show("median sign loss: low", per_model(lambda m: lo(sign(m)).median()))
show("median sign loss: others", per_model(lambda m: rest(sign(m)).median()))


def recall_loss(m):
    x = a[(a.model == m) & (a.scheme == "binary")].set_index("lang"); f = a[(a.model == m) & (a.scheme == "fp32")].set_index("lang")
    return 100 * (f.recall_at_10 - x.recall_at_10.reindex(f.index)) / f.recall_at_10


show("recall@10 loss, sign: English", per_model(lambda m: recall_loss(m)["eng_Latn"]))
show("recall@10 loss, sign: all", per_model(lambda m: recall_loss(m).mean()))
show("recall@10 loss, sign: low", per_model(lambda m: lo(recall_loss(m)).mean()))
show("BGE-M3 worst language, sign", sign("bge-m3").max())
show("E5-small best language, sign", sign("me5-small").min())

print("\n3.2 Margin and anisotropy")
fp = {m: a[(a.model == m) & (a.scheme == "fp32")].set_index("lang") for m in M}
show("Spearman margin vs sign loss", per_model(lambda m: spearmanr(fp[m].margin_mean, sign(m).reindex(fp[m].index))[0]))
show("Spearman float32 nDCG vs sign loss", per_model(lambda m: spearmanr(fp[m].ndcg_at_10, sign(m).reindex(fp[m].index))[0]))
show("mean margin low", per_model(lambda m: lo(fp[m].margin_mean).mean()))
show("mean margin others", per_model(lambda m: rest(fp[m].margin_mean).mean()))
corr = pd.DataFrame({m: fp[m].margin_mean for m in M}).corr("spearman").values[np.triu_indices(5, 1)]
show("Spearman of margins between models, min and max", (round(corr.min(), 3), round(corr.max(), 3)))
for m in M:
    sel = fp[m][(fp[m].margin_mean >= .04) & (fp[m].margin_mean <= .06)].index
    show(f"sign loss at margin 0.04-0.06, {m}", (len(sel), round(sign(m)[sel].min(), 1), round(sign(m)[sel].max(), 1)))
s = a[a.scheme == "binary"].copy()
show("pairs; R2 margin; R2 margin * c", (len(s), round(smf.ols("drop_pct~margin_mean", s).fit().rsquared, 3), round(smf.ols("drop_pct~margin_mean*doc_cos_mean", s).fit().rsquared, 3)))
show("mean c per model", per_model(lambda m: fp[m].doc_cos_mean.mean()))
show("E5-small / E5-large sign loss", sign("me5-small").mean() / sign("me5-large").mean())
t = s.dropna(subset=["joshi_class"])
f = smf.ols("drop_pct~C(model)+C(model):margin_mean+joshi_class+fertility+latin", t).fit(cov_type="cluster", cov_kwds={"groups": t.lang.astype("category").cat.codes})
show("pairs with class; R2 with a margin slope per model", (len(t), round(f.rsquared, 3)))
show("coefficients", f.params[["joshi_class", "fertility", "latin"]].to_dict()); show("p values", f.pvalues[["joshi_class", "fertility", "latin"]].to_dict())
f = smf.ols("drop_pct~C(model)+C(model):margin_mean+C(model):joshi_class+C(model):fertility+C(model):latin-1", t).fit(cov_type="cluster", cov_kwds={"groups": t.lang.astype("category").cat.codes})
for term in ["joshi_class", "fertility", "latin"]:
    show(f"{term} per model", f.params.filter(like=term).rename(lambda k: k.split("[")[1].split("]")[0].replace("T.", "")).to_dict())
pairs = {"arb_Latn": "arb_Arab", "ben_Latn": "ben_Beng", "hin_Latn": "hin_Deva", "npi_Latn": "npi_Deva", "sin_Latn": "sin_Sinh", "urd_Latn": "urd_Arab"}
show("romanized variants losing more than native (of 6)", per_model(lambda m: sum(sign(m)[k] > sign(m)[v] for k, v in pairs.items())))
g = sign("bge-m3"); show("BGE-M3 native range, romanized range", (round(min(g[v] for v in pairs.values()), 1), round(max(g[v] for v in pairs.values()), 1), round(min(g[k] for k in pairs), 1), round(max(g[k] for k in pairs), 1)))

print("\n3.3 Calibration")
UNC = ["fp8_e4m3", "int8", "int4", "int4_p1", "int2", "int2_p1", "int2_p5", "binary_cal"]
show("E5 mean loss by format", {f"{m} {sc}": (sign(m) if sc == "binary" else loss(m, sc, "english")).mean() for m in E5 for sc in ["binary", "binary_cal", "int2", "int2_p5"]})
d = []
for m in M:
    for sc in UNC:
        e, x = loss(m, sc, "english"), loss(m, sc, "mixed")
        if len(x):
            d += [abs(e.mean() - x.mean()), abs(lo(e).mean() - lo(x).mean()), abs(rest(e).mean() - rest(x).mean())]
show("uncentered, English to mixed, largest change of a group mean", max(d))
rec = [100 * (loss(m, sc, "english").mean() - loss(m, sc, "target").mean()) / loss(m, sc, "english").mean() for m in E5 for sc in ["int2", "int2_p5", "binary_cal"]]
show("E5 loss recovered by 1388 target-language texts, 1-2 bits (%), min and max", (round(min(rec), 1), round(max(rec), 1)))
h = pd.read_csv("results/heldout_calibration_me5-small.csv")
gp = h.groupby(["scheme", "calib", "seed"], dropna=False).apply(lambda x: pd.Series({"all": x.loss.mean(), "low": x[x.lang.isin(low)].loss.mean()}), include_groups=False)
ag = gp.groupby(level=["scheme", "calib"]).agg(["mean", "std"])
print(ag.round(2).to_string())
for col in ["all", "low"]:
    show(f"unrelated Wikipedia vs Belebele English, change of {col} by scheme", {sc: ag.loc[(sc, "wiki_en"), (col, "mean")] - ag.loc[(sc, "bele_en"), (col, "mean")] for sc in ag.index.get_level_values(0).unique()})
show("largest standard deviation over Wikipedia draws", ag.xs("wiki_en", level="calib").xs("std", axis=1, level=1).max().max())
show("int2 / int2-clip mean loss", per_model(lambda m: loss(m, "int2", "english").mean() / loss(m, "int2_p5", "english").mean()))
show("low-resource int2 / int2-clip", per_model(lambda m: lo(loss(m, "int2", "english")).mean() / lo(loss(m, "int2_p5", "english")).mean()))

print("\n3.4 Centering")
show("sign mean", per_model(lambda m: plain(m).loss.mean())); show("sign low", per_model(lambda m: lo(plain(m).loss).mean()))
show("centered mean", per_model(lambda m: centered(m).loss.mean())); show("centered low", per_model(lambda m: lo(centered(m).loss).mean()))
show("share of the sign loss recovered, Belebele (%)", per_model(lambda m: 100 * (plain(m).loss.mean() - centered(m).loss.mean()) / plain(m).loss.mean()))
show("change of mean sign loss", per_model(lambda m: plain(m).loss.mean() - centered(m).loss.mean()))
show("languages over 4% after centering", per_model(lambda m: (centered(m).loss > 4).sum()))
show("worst language after centering", per_model(lambda m: centered(m).loss.max()))
show("E5-small float32 nDCG before, after", (round(X["me5-small"][X["me5-small"].method == "check_fp32"].base_ndcg.mean(), 3), round(centered("me5-small").base_ndcg_t.mean(), 3)))
show("E5-small sign nDCG before, after", (round(plain("me5-small").ndcg.mean(), 3), round(centered("me5-small").ndcg.mean(), 3)))
show("relative to centered float32: all", per_model(lambda m: centered(m).loss_vs_t.mean()))
show("relative to centered float32: low", per_model(lambda m: lo(centered(m).loss_vs_t).mean()))
show("relative to centered float32: English", per_model(lambda m: centered(m).loss_vs_t["eng_Latn"]))
for n in [16, 1388]:
    show(f"mean loss, mean of {n} English texts", per_model(lambda m: X[m][(X[m].method == "center_sign") & (X[m].n == n)].groupby("seed").loss.mean().mean()))
show("E5-small, mean from unrelated Wikipedia text", ag.loc[("centered_sign", "wiki_en"), ("all", "mean")])
wc = {m: W[(W.model == m) & (W["transform"] == "center")] for m in M}
show("mixed mean, sign loss", per_model(lambda m: wc[m][wc[m].calib == "mixed"].binary_loss_vs_orig.mean()))
show("low-resource shift, English to mixed mean", per_model(lambda m: wc[m][(wc[m].calib == "english") & wc[m].lang.isin(low)].binary_loss_vs_orig.mean() - wc[m][(wc[m].calib == "mixed") & wc[m].lang.isin(low)].binary_loss_vs_orig.mean()))
d = plain("me5-large").loss - sign("me5-large").reindex(plain("me5-large").index)
show("E5-large, recomputed vs stored: languages over 0.5 points", (d.abs() > .5).sum())
dn = (plain("me5-large").ndcg - plain("me5-large").stored_ndcg).abs(); show("E5-large, largest change of sign nDCG", (dn.idxmax(), round(dn.max(), 3)))

print("\n3.4 Both sides must be centered")
show("mean-threshold, query uncentered", per_model(lambda m: loss(m, "binary_cal", "english").mean())); show("plain sign", per_model(lambda m: sign(m).mean()))
o = pd.read_csv("results/mechanism_onesided_me5-small.csv"); show("E5-small: bits only, query only, both", o[["loss_bits_only", "loss_query_only", "loss_both"]].mean().to_dict())
b = pd.read_csv("results/mechanism_bits_me5-small.csv"); show("E5-small bits shared, offset sd, query part sd", b[["bits_shared_sign", "bits_shared_centered", "offset_sd", "query_part_sd"]].mean().to_dict())

print("\n3.4 Stepwise removal")
g = pd.read_csv("results/mechanism_me5-small.csv").groupby("alpha").agg(c=("doc_cos", "mean"), margin=("margin", "mean"), noise=("noise", "mean"), loss=("loss_vs_orig", "mean"))
g["margin_over_noise"] = g.margin / g.noise; print(g.round(3).to_string())
q = pd.read_csv("results/mechanism_queries_me5-small.csv")
for al in [0, 1]:
    x = q[q.alpha == al]; show(f"alpha={al}: lost first rank, ratio<0.25 and ratio>2 (%)", (round(100 * x[x.ratio < .25].lost.mean(), 1), round(100 * x[x.ratio > 2].lost.mean(), 1)))

print("\n3.5 Other methods and the larger corpus")
p = rows("me5-small", "pq"); show("E5-small PQ English, all, low", (round(p.loss["eng_Latn"], 1), round(p.loss.mean(), 1), round(lo(p.loss).mean(), 1)))
show("PQ, English to all-language training, change of mean", per_model(lambda m: rows(m, "pq", "multi10k").loss.mean() - rows(m, "pq").loss.mean()))
show("centered PQ mean", per_model(lambda m: rows(m, "pq_c").loss.mean()))
for m in M:
    show(f"rescoring with int8, sign, {m}, loss by candidates", R[(R.model == m) & (R.scheme == "binary") & (R.rescore == "int8")].groupby("k").loss.mean().to_dict())
show("compressed query minus float32 query, sign", per_model(lambda m: loss(m, "binary", frame=sym).mean() - sign(m).mean()))
show("compressed query / float32 query, sign", per_model(lambda m: loss(m, "binary", frame=sym).mean() / sign(m).mean()))

print("\n3.5 Wikipedia task")
for m in M:
    w = pd.read_csv(f"results/wikimulti_{m}.csv"); w = w[w.qmode == "asym"]; sg = 100 * w[w.scheme == "binary"].set_index("lang").rel_drop; f_ = w[w.scheme == "fp32"].set_index("lang")
    show(f"{m}: languages, sign mean, English, Spearman margin", (len(sg), round(sg.mean(), 1), round(sg["eng_Latn"], 1), round(spearmanr(f_.margin_mean, sg.reindex(f_.index))[0], 2)))
for m in E5:
    x = XW[m]; sg, c = plain(m, XW), centered(m, XW); mg = x[x.method == "sign_margin"].set_index("lang"); f_ = x[x.method == "check_fp32"].set_index("lang")
    show(f"{m}: margin fold", c.margin_t.mean() / mg.margin.mean())
    show(f"{m}: sign mean, centered mean, recovered %", (round(sg.loss.mean(), 1), round(c.loss.mean(), 1), round(100 * (sg.loss.mean() - c.loss.mean()) / sg.loss.mean())))
    show(f"{m}: worst language sign, centered", (round(sg.loss.max(), 1), round(c.loss.max(), 1)))
    show(f"{m}: float32 change (%), loss relative to centered float32", (round(100 * (f_.base_ndcg.mean() - c.base_ndcg_t.mean()) / f_.base_ndcg.mean(), 2), round(c.loss_vs_t.mean(), 2)))
