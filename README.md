# Compressed embeddings fail unevenly across languages

Code and per-language results for *Compressed Embeddings Fail Unevenly Across Languages: Score Margins, Anisotropy and Mean Removal*.

The study measures what post-hoc compression of stored text embeddings (float16, float8, int8, int4, int2, binary) costs retrieval quality in 122 languages with parallel content, for five multilingual embedding models, and relates the loss to two float32 statistics: the score margin between the relevant document and the best distractor (within a model) and the anisotropy of the embedding space (across models).

## Main results

All on Belebele unless stated. Loss is the relative drop of nDCG@10 from float32, averaged over languages.

- int8 costs under 0.6% and int4 at most 1.9% on average in every model, although single languages lose up to 10% at 4 bits.
- Below 4 bits the loss is uneven in every model. English loses less than most languages, not all of them, and the 47 low-resource languages lose more than the other 75. On the Wikipedia task English is not always best.
- How much is lost depends on the model. Under sign binarization the mean loss is 27-59% for multilingual-E5 and 2-3% for BGE-M3 and Qwen3-Embedding; E5-large has the same 1024 dimensions as those two.
- Within a model the float32 margin orders the languages (Spearman -0.81 to -0.98 for sign binarization), and float32 nDCG@10 does so almost as well. Across models the loss tracks anisotropy; this relation is descriptive and rests on five models.
- Without centering, the calibration language matters little (group means change by at most 3.7 points). Percentile-clipped ranges halve the 2-bit loss.
- Subtracting one English mean vector from documents *and* queries cuts the mean 1-bit loss of the E5 models to 1.8-3.9%, recovering over 90% of it. The effect is smaller elsewhere: 39% and 69% for BGE-M3 and Qwen3, and 82%, 48% and 25% for the E5 models on the Wikipedia task.
- Centering does not remove the unevenness. Single languages still lose up to 20%, 74 of 122 languages still lose over 4% with E5-small, and low-resource languages still lose more than English relative to the centered float32 quality. After centering the calibration data does matter: a mixed-language mean shifts low-resource losses by 3.2-5.5 points for the E5 models.

## Repository layout

```
src/                 pipeline (see below)
results/             per-language results as CSV (the numbers, figure and tables of the paper are computed from these)
cloud/               scripts to compute the embeddings (and the equal-storage experiments) on a GPU machine
data/lang_meta.csv   per-language metadata: script, Joshi et al. (2020) resource class, Wikipedia size
data/lang2tax.txt    resource classes from Joshi et al. (2020), https://microsoft.github.io/linguisticdiversity/
run_all.sh           the full pipeline in order
```

### Pipeline (`src/`)

| script | role |
|---|---|
| `data.py` | loads the two MTEB tasks (BelebeleRetrieval, WikipediaRetrievalMultilingual) into corpus / queries / relevance labels |
| `models.py` | the five models and their query/document prefixes |
| `embed.py` | computes float32 embeddings once per model, task and language and stores them as `.npz` |
| `quant.py` | the storage formats: float16, float8, int8/int4/int2 with min-max or percentile ranges, binary with zero or mean threshold |
| `metrics.py` | nDCG@10, recall@10, MRR@10 (identical to pytrec_eval for one relevant document per query) |
| `evaluate.py` | applies every format x calibration source x query mode to the stored embeddings; writes `results/<task>_<model>.csv` |
| `whiten_rescore.py` | centering / ZCA whitening before compression, and top-k rescoring with an int8 corpus |
| `extra_experiments.py` | product quantization and PCA at equal storage, centering sample size, centering on the Wikipedia task; writes `results/extra_<task>_<model>.csv` (needs re-computed embeddings) |
| `analyze.py`, `analyze_wikimulti.py` | statistics, regressions, merged results and exploratory plots (`analyze.py` also needs the embeddings for tokenizer fertility) |
| `make_paper_assets.py` | Figure 1 and Tables 1-2 of the paper, written to `paper/` from the result files |
| `paper_numbers.py` | recomputes the numbers in the prose of the paper, section by section, from the result files |
| `mechanism.py` | score perturbation, ranking flips, graded centering, shared bits and one-sided centering on the stored E5-small embeddings; writes `results/mechanism*_me5-small.csv` |
| `heldout_calibration.py` | calibration on English Wikipedia passages (no content overlap with Belebele) and multi-seed mixed calibration, E5-small; writes `results/heldout_calibration_me5-small.csv` |
| `lang_meta.py` | builds `data/lang_meta.csv` |

## Reproducing the paper

Requirements: Python 3.12, `pip install -r requirements.txt` (torch, sentence-transformers, mteb, datasets, numpy, pandas, scipy, statsmodels, matplotlib, faiss-cpu).

1. **Embeddings** (GPU; about 6 T4-hours in total, 2.9 GB of `.npz` files for both tasks). On any CUDA machine:
   ```
   bash cloud/run_cloud.sh            # BGE-M3 and Qwen3 on Belebele
   bash cloud/run_cloud_wikimulti.sh  # all models on the Wikipedia task
   bash cloud/run_cloud_me5large.sh   # multilingual-e5-large on both tasks
   python src/embed.py --task belebele --model me5-small   # e5-small and e5-base run on a laptop
   python src/embed.py --task belebele --model me5-base
   bash cloud/run_cloud_extra.sh      # re-embeds and runs src/extra_experiments.py (equal-storage comparison, centering sample size, Wikipedia centering)
   ```
   Datasets are downloaded from the Hugging Face Hub (`mteb/belebele`, `mteb/WikipediaRetrievalMultilingual`) at pinned revisions.
2. **Evaluation and analysis** (laptop, minutes): the remaining steps of `run_all.sh`.

Without step 1, from the released files alone: `python src/make_paper_assets.py` and `python src/paper_numbers.py` read only `results/*.csv` and `data/lang_meta.csv`, and `analyze_wikimulti.py` reads only `results/`.

The equal-storage, centering sample-size and Wikipedia centering results use re-computed embeddings; the other results use the original embeddings. Re-embedding changed 18 E5-large languages by over 0.5 points (Chinese by 0.16 sign nDCG@10), for a cause not identified.

## Result files

`results/<model>.csv` (Belebele) and `results/wikimulti_<model>.csv` have one row per language x format x calibration source x query mode with columns

- `scheme`, `bits`, `calib` (`english`, `mixed`, `mixed_lat`, `target64`, `target`, or `none`), `qmode` (`asym` = query float32, `sym` = query compressed)
- `ndcg_at_10`, `recall_at_10`, `mrr_at_10`, `base_ndcg` (float32), `rel_drop` (relative loss, fraction)
- float32 geometry per language: `margin_mean`, `margin_p25`, `doc_cos_mean`, `doc_cos_std`, `dim_var_cv`
- `clip_frac` (fraction of values outside the calibration range, integer formats)

`results/all_results_merged.csv` joins these with the language metadata and tokenizer fertility. `results/whitening.csv` and `results/rescoring.csv` hold the centering/whitening and rescoring experiments; `results/extra_<task>_<model>.csv` hold the equal-storage, centering sample-size and Wikipedia centering experiments (one row per language x method x budget x calibration; `extra_belebele_me5-small_mac.csv` is a cross-check run on different hardware). `results/summary_*.csv` are the aggregated tables.

## Data and model sources

- Belebele: Bandarkar et al. (2024), CC BY-SA 4.0, via MTEB (`mteb/belebele`, revision `979a211`).
- Wikipedia retrieval: MTEB `WikipediaRetrievalMultilingual` (revision `5f6c91d`).
- Resource classes: Joshi et al. (2020), *The State and Fate of Linguistic Diversity and Inclusion in the NLP World*.
- Models: `intfloat/multilingual-e5-{small,base,large}`, `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B`, used at their released checkpoints.

## License

MIT (see `LICENSE`). The datasets and models keep their own licenses.
