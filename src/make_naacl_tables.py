"""Table for the NAACL version: compression at equal storage (1 bit per dimension = d/8 bytes per vector).

Reads results/extra_belebele_<model>.csv (from src/extra_experiments.py); writes paper_acl/tab_matched.tex.
"""
import pandas as pd
M = ["me5-small", "me5-base", "me5-large", "bge-m3", "qwen3-0.6b"]
NICE = {"me5-small": "E5-small", "me5-base": "E5-base", "me5-large": "E5-large", "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
meta = pd.read_csv("data/lang_meta.csv").set_index("lang")
COLS = [("check_sign", "-", "sign"), ("center_sign", "english", "centered sign"), ("pq", "english", "PQ"),
        ("pq_c", "english", "centered PQ"), ("pca8", "english", "PCA + int8")]
L = ["\\begin{table*}[t]\\centering\\footnotesize", "\\begin{tabular}{l" + "rr" * len(COLS) + "}", "\\toprule",
     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{c[2]}}}" for c in COLS) + " \\\\",
     " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(COLS))),
     "model & " + " & ".join(["all & low"] * len(COLS)) + " \\\\", "\\midrule"]
for m in M:
    x = pd.read_csv(f"results/extra_belebele_{m}.csv").join(meta[["joshi_class"]], on="lang")
    cells = []
    for meth, cal, _ in COLS:
        s = x[(x.method == meth) & (x.budget == "1bit")]
        if meth == "center_sign":
            s = s[s.n == 1388]
        elif cal != "-":
            s = s[s.calib == cal]
        cells += [s.loss.mean(), s[s.joshi_class <= 1].loss.mean()]
    L.append(NICE[m] + " & " + " & ".join(f"{0.0 if abs(v) < 0.05 else v:.1f}" for v in cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Loss (\\%) relative to the original float32 nDCG@10 at the storage of 1 bit per dimension ($d/8$ bytes per vector): "
      "sign binarization, sign after the centering transform (documents and queries), product quantization with $d/8$ 8-bit sub-quantizers (PQ), "
      "PQ after centering, and PCA to $d/8$ dimensions stored at int8. All fitted on 1388 English vectors; query float32. "
      "Computed on re-embedded vectors (Appendix~\\ref{app:repro}).}\\label{tab:matched}",
      "\\end{table*}"]
open("paper_acl/tab_matched.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
