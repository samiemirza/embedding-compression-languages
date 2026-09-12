# Compressed embeddings fail unevenly across languages

Code, results and manuscript for *Compressed embeddings fail unevenly across languages: the score margin, not the calibration language, decides* (under review).

The study measures what post-hoc compression of stored text embeddings (float16, float8, int8, int4, int2, binary) costs retrieval quality in 122 languages with identical content, for five multilingual embedding models, and explains the loss with a single float32 quantity: the score margin between the relevant document and the best distractor.

## Main results

- At 8 and 4 bits nothing is lost in any language. Below that, English loses least and low-resource languages most, in every model.
- How much is lost is set by the model, not the bit width or the dimensionality: the multilingual-E5 models lose 7-71% of nDCG@10 under 1-bit storage, BGE-M3 and Qwen3-Embedding under 5% at the same 1024 dimensions.
- The float32 margin orders the languages within every model (Spearman -0.81 to -0.98) and explains three quarters of the variance across models and languages.
- The calibration language is not the cause (at most 3.2 points). Percentile-clipped ranges halve the 2-bit loss, and subtracting the calibration mean before compression removes most of the E5 loss in every language.

## Repository layout

```
src/                 pipeline (see below)
results/             per-language results as CSV (everything in the paper is computed from these)
paper_tmlr/          manuscript (main.tex), generated tables (tab_*.tex) and figures (figs/)
cloud/               scripts to compute the embeddings on a GPU machine
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
| `quant.py` | the storage formats: float16, float8, int8/int4/int2 with min-max or percentile ranges, binary with zero or centred threshold |
| `metrics.py` | nDCG@10, recall@10, MRR@10 (identical to pytrec_eval for one relevant document per query) |
| `evaluate.py` | applies every format x calibration source x query mode to the stored embeddings; writes `results/<task>_<model>.csv` |
| `whiten_rescore.py` | centring / ZCA whitening before compression, and top-k rescoring with an int8 corpus |
| `analyze.py`, `analyze_wikimulti.py` | statistics, regressions and figures |
| `make_draft_tables.py`, `make_tmlr_tables.py`, `make_wr_tables.py` | every table in the manuscript |
| `lang_meta.py` | builds `data/lang_meta.csv` |

## Reproducing the paper

Requirements: Python 3.12, `pip install -r requirements.txt` (torch, sentence-transformers, mteb, datasets, numpy, pandas, scipy, statsmodels, matplotlib).

1. **Embeddings** (GPU; about 6 T4-hours in total, 2.9 GB of `.npz` files for both tasks). On any CUDA machine:
   ```
   bash cloud/run_cloud.sh            # BGE-M3 and Qwen3 on Belebele
   bash cloud/run_cloud_wikimulti.sh  # all models on the Wikipedia task
   bash cloud/run_cloud_me5large.sh   # multilingual-e5-large on both tasks
   python src/embed.py --task belebele --model me5-small   # e5-small and e5-base run on a laptop
   python src/embed.py --task belebele --model me5-base
   ```
   Datasets are downloaded from the Hugging Face Hub (`mteb/belebele`, `mteb/WikipediaRetrievalMultilingual`) at pinned revisions.
2. **Evaluation and analysis** (laptop, minutes): the remaining steps of `run_all.sh`.

Steps 2 onward can be run without step 1 from the released `results/` files: `analyze.py`, `analyze_wikimulti.py` and the `make_*_tables.py` scripts read only `results/*.csv` and `data/lang_meta.csv`.

## Result files

`results/<model>.csv` (Belebele) and `results/wikimulti_<model>.csv` have one row per language x format x calibration source x query mode with columns

- `scheme`, `bits`, `calib` (`english`, `mixed`, `mixed_lat`, `target64`, `target`, or `none`), `qmode` (`asym` = query float32, `sym` = query compressed)
- `ndcg_at_10`, `recall_at_10`, `mrr_at_10`, `base_ndcg` (float32), `rel_drop` (relative loss, fraction)
- float32 geometry per language: `margin_mean`, `margin_p25`, `doc_cos_mean`, `doc_cos_std`, `dim_var_cv`
- `clip_frac` (fraction of values outside the calibration range, integer formats)

`results/all_results_merged.csv` joins these with the language metadata and tokenizer fertility. `results/whitening.csv` and `results/rescoring.csv` hold the centring/whitening and rescoring experiments. `results/summary_*.csv` are the aggregated tables.

## Data and model sources

- Belebele: Bandarkar et al. (2024), CC BY-SA 4.0, via MTEB (`mteb/belebele`, revision `979a211`).
- Wikipedia retrieval: MTEB `WikipediaRetrievalMultilingual` (revision `5f6c91d`).
- Resource classes: Joshi et al. (2020), *The State and Fate of Linguistic Diversity and Inclusion in the NLP World*.
- Models: `intfloat/multilingual-e5-{small,base,large}`, `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B`, used at their released checkpoints.

## License

MIT (see `LICENSE`). The datasets and models keep their own licenses.
