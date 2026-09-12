#!/bin/bash
# Run the two remaining models on any CUDA box (Kaggle/Colab T4, RunPod, Vast.ai, Lambda).
# Usage: bash cloud/run_cloud.sh   (from the repo root; needs ~3 GB disk for weights + 1.4 GB for outputs)
set -e
pip install -q "torch" sentence-transformers mteb datasets numpy
# bge-m3 in fp32 to match the local e5 runs; qwen3 is fp16 by registry (same as the local test).
python src/embed.py --model bge-m3 --batch 64
python src/embed.py --model qwen3-0.6b --batch 32
tar czf embeddings_cloud.tgz embeddings/bge-m3 embeddings/qwen3-0.6b
ls -lh embeddings_cloud.tgz
# Bring embeddings_cloud.tgz back to the Mac, extract in the repo root, then:
#   python src/evaluate.py --model bge-m3 && python src/evaluate.py --model qwen3-0.6b && python src/analyze.py
