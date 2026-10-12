#!/bin/bash
# Full reproduction. Step 1 needs a GPU (about 6 T4-hours in total); steps 2-4 run on a laptop in minutes.
set -e
MODELS="me5-small me5-base me5-large bge-m3 qwen3-0.6b"
# 1. embeddings (float32 .npz per language) -- or run cloud/run_cloud*.sh on a GPU machine
for m in $MODELS; do
  python src/embed.py --task belebele  --model $m --batch 64
  python src/embed.py --task wikimulti --model $m --batch 64
done
# 2. apply every format / calibration source / query mode to the stored embeddings
for m in $MODELS; do
  python src/evaluate.py --task belebele  --model $m
  python src/evaluate.py --task wikimulti --model $m
done
python src/whiten_rescore.py
# 3. analysis: statistics, figures
python src/analyze.py            # -> results/summary_*.csv, build/tab*.tex, build/figs/fig1-3
python src/analyze_wikimulti.py  # -> results/summary_wikimulti.csv, build/tab_wiki.tex
python src/extra_experiments.py --task belebele --model me5-small   # per model and task; see cloud/run_cloud_naacl.sh
# 4. tables for the manuscript
python src/make_draft_tables.py  # -> build/tab_drop.tex, tab_geometry.tex, tab_calib.tex
python src/make_appendix_tables.py  # -> build/tab_perlang.tex, tab_sym.tex, tab_calib_full.tex
python src/make_wr_tables.py     # -> build/tab_whiten.tex, tab_rescore.tex, build/figs/fig4
python src/make_naacl_tables.py  # -> paper_acl/tab_matched.tex
python src/make_naacl_figs.py    # -> paper_acl/figs/fig1-3
