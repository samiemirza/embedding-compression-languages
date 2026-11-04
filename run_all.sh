#!/bin/bash
# Full reproduction. Step 1 needs a GPU (about 6 T4-hours in total); steps 2-4 run on a laptop.
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
python src/mechanism.py             # E5-small only
python src/heldout_calibration.py   # E5-small only
for t in belebele wikimulti; do for m in me5-small me5-base me5-large; do python src/extra_experiments.py --task $t --model $m; done; done
for m in bge-m3 qwen3-0.6b; do python src/extra_experiments.py --task belebele --model $m; done
# 3. analysis: statistics
python src/analyze.py            # -> results/summary_*.csv, results/all_results_merged.csv
python src/analyze_wikimulti.py  # -> results/summary_wikimulti.csv
# 4. the paper's figure, tables and numbers (these two read only results/ and data/)
python src/make_paper_assets.py  # -> paper/fig_margin.pdf, tab_loss.tex, tab_storage.tex
python src/paper_numbers.py      # prints the numbers of the paper, section by section
