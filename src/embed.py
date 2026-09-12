"""Embed all 122 Belebele languages with one model; save float32 embeddings.

Usage: python src/embed.py --model me5-small [--langs eng_Latn,swh_Latn] [--purge]
Output: embeddings/<model>/<lang>.npz with keys docs (488,d), queries (900,d), qrels (900,)
        embeddings/<model>/fertility.csv with tokens per passage per language
"""
from __future__ import annotations
import argparse, csv, os, shutil, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import task_languages, load_task_lang
from models import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--task", default="belebele", choices=["belebele", "wikimulti"])
    ap.add_argument("--langs", default=None, help="comma list; default all 122")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--purge", action="store_true", help="delete HF model weights afterwards")
    ap.add_argument("--fp16", action="store_true", help="run the encoder in float16 (GPU); embeddings still stored as float32")
    a = ap.parse_args()

    hf_id, qp, dp, trc, *rest = MODELS[a.model]
    mk = dict(rest[0]) if rest else {}
    if mk.get("torch_dtype") == "float16" or a.fp16:
        mk["torch_dtype"] = torch.float16
    out_dir = os.path.join("embeddings", a.model) if a.task == "belebele" else os.path.join("embeddings", a.task, a.model)
    os.makedirs(out_dir, exist_ok=True)
    langs = a.langs.split(",") if a.langs else task_languages(a.task)

    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(hf_id, device=device, trust_remote_code=trc, model_kwargs=mk)
    model.max_seq_length = a.max_len
    tok = model.tokenizer
    print(f"{a.model}: {hf_id} dim={model.get_sentence_embedding_dimension()} device={device}", flush=True)

    fert_path = os.path.join(out_dir, "fertility.csv")
    fert = {}
    if os.path.exists(fert_path):
        with open(fert_path) as f:
            fert = {r["lang"]: r for r in csv.DictReader(f)}

    for i, lang in enumerate(langs):
        path = os.path.join(out_dir, f"{lang}.npz")
        if os.path.exists(path) and lang in fert:
            continue
        t0 = time.time()
        d = load_task_lang(a.task, lang)
        docs = model.encode([dp + x for x in d.docs], batch_size=a.batch, show_progress_bar=False, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
        qs = model.encode([qp + x for x in d.queries], batch_size=a.batch, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.savez(path, docs=docs, queries=qs, qrels=d.qrels)
        # tokenizer fertility: tokens per passage / per query (no prefix), chars per passage
        n_doc_tok = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in d.docs]
        n_q_tok = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in d.queries]
        n_unk = sum(t == tok.unk_token_id for x in d.docs for t in tok(x, add_special_tokens=False)["input_ids"]) if tok.unk_token_id is not None else 0
        fert[lang] = dict(lang=lang, doc_tokens=np.mean(n_doc_tok), query_tokens=np.mean(n_q_tok),
                          doc_chars=np.mean([len(x) for x in d.docs]),
                          doc_words=np.mean([len(x.split()) for x in d.docs]),
                          doc_trunc_frac=np.mean([n > a.max_len - 2 for n in n_doc_tok]),
                          unk_frac=n_unk / max(1, sum(n_doc_tok)))
        with open(fert_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(fert.values())).keys()))
            w.writeheader(); w.writerows(fert.values())
        print(f"[{i+1}/{len(langs)}] {lang} {time.time()-t0:.1f}s doc_tok={fert[lang]['doc_tokens']:.0f}", flush=True)

    if a.purge:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == hf_id:
                shutil.rmtree(repo.repo_path, ignore_errors=True)
                print("purged", repo.repo_path)


if __name__ == "__main__":
    main()
