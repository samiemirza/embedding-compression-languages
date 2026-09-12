#!/bin/bash
# Second task (WikipediaRetrievalMultilingual, 16 languages) for all four models on any CUDA box.
# Usage from the repo root: bash cloud/run_cloud_wikimulti.sh
set -e
pip install -q sentence-transformers mteb datasets numpy
python src/embed.py --task wikimulti --model me5-small  --batch 128
python src/embed.py --task wikimulti --model me5-base   --batch 64
python src/embed.py --task wikimulti --model bge-m3     --batch 64
python src/embed.py --task wikimulti --model qwen3-0.6b --batch 32
tar cf embeddings_wikimulti.tar embeddings/wikimulti
ls -lh embeddings_wikimulti.tar
# Back on the Mac (repo root):  tar xf embeddings_wikimulti.tar
#   for m in me5-small me5-base bge-m3 qwen3-0.6b; do python src/evaluate.py --task wikimulti --model $m; done
