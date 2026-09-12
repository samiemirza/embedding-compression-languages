#!/bin/bash
# multilingual-e5-large (560M, 1024-d) on both tasks: Belebele (122 languages) and Wikipedia (16 languages).
# All dataset files are shipped in data/, so only the model weights are downloaded.
set -e
pip install -q sentence-transformers mteb datasets numpy pandas pyarrow
python src/embed.py --task belebele  --model me5-large --batch 64
python src/embed.py --task wikimulti --model me5-large --batch 64
tar cf embeddings_me5large.tar embeddings/me5-large embeddings/wikimulti/me5-large
ls -lh embeddings_me5large.tar
# Back on the Mac (repo root):  tar xf embeddings_me5large.tar
#   python src/evaluate.py --model me5-large && python src/evaluate.py --task wikimulti --model me5-large
#   python src/analyze.py && python src/analyze_wikimulti.py
