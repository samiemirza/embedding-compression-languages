#!/bin/bash
# Equal-storage and centering runs on a Kaggle/Colab T4: re-embed, run src/extra_experiments.py, keep only the small result CSVs.
# Usage from the repo root: bash cloud/run_cloud_extra.sh   Results: $OUT (default /kaggle/working/extra_results)
# TEST=1 runs E5-small on a few languages of each task and skips package installs (local check before uploading).
set -o pipefail
OUT=${OUT:-/kaggle/working/extra_results}; mkdir -p "$OUT"
export HF_HOME=${HF_HOME:-/tmp/hf}
if [ "$TEST" != 1 ]; then
  python -m pip install -q faiss-cpu
  python -c "import sentence_transformers" 2>/dev/null || python -m pip install -q sentence-transformers
  python -c "import transformers; from packaging.version import Version as V; assert V(transformers.__version__) >= V('4.51')" 2>/dev/null \
    || python -m pip install -q "transformers>=4.51"
fi
# preflight: stop within seconds if anything the steps need is missing
python - <<'PY' 2>&1 | tee "$OUT/run.log" || { echo "PREFLIGHT FAILED"; exit 1; }
import sys; sys.path.insert(0, "src")
import torch, transformers, sentence_transformers, faiss, pandas
from data import task_languages, english_code, WIKI_CODE
assert len(task_languages("belebele")) == 122 and len(task_languages("wikimulti")) == 16
import os; assert os.path.exists("results/me5-small.csv") and os.path.exists("results/wikimulti_me5-small.csv")
print("preflight ok | torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__)
PY
step() {  # task model batch [langs]
  echo "=== $1 $2 start $(date)" | tee -a "$OUT/run.log"
  python -W ignore src/embed.py --task $1 --model $2 --batch $3 ${4:+--langs $4} >> "$OUT/run.log" 2>&1 \
    && python -W ignore src/extra_experiments.py --task $1 --model $2 2>&1 | grep -v "WARNING clustering" >> "$OUT/run.log" \
    && cp results/extra_$1_$2.csv "$OUT/"
  rc=$?
  echo "=== $1 $2 done rc=$rc $(date)" | tee -a "$OUT/run.log"
  rm -rf embeddings/$2 embeddings/wikimulti/$2 "$HF_HOME"/hub/models--*   # free disk
}
if [ "$TEST" = 1 ]; then
  step wikimulti me5-small 128 en,bn
  step belebele  me5-small 128 eng_Latn,afr_Latn,ben_Beng
else
  step wikimulti me5-small  128
  step wikimulti me5-base   64
  step wikimulti me5-large  64
  step belebele  me5-base   64
  step belebele  me5-large  64
  step belebele  bge-m3     64
  step belebele  qwen3-0.6b 32
  step belebele  me5-small  128   # already run on the Mac; repeated as a cross-check
fi
echo "=== ALL DONE $(date)" | tee -a "$OUT/run.log"
grep "done rc=" "$OUT/run.log"
(cd "$OUT" && tar czf ../extra_results.tgz .) && ls -lh "$OUT"
