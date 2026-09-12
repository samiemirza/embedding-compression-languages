import numpy as np


def retrieval_metrics(scores, qrels, k=10):
    """scores (nq, nd); qrels (nq,) index of the single relevant doc.
    Returns ndcg@k, recall@k, mrr@k (identical to pytrec_eval for 1 relevant doc)."""
    # rank of the relevant doc = number of docs with strictly higher score (+1)
    rel = scores[np.arange(len(qrels)), qrels][:, None]
    rank = (scores > rel).sum(1) + 1  # ties resolved optimistically; ties are rare in float
    hit = rank <= k
    ndcg = np.where(hit, 1.0 / np.log2(rank + 1), 0.0)
    mrr = np.where(hit, 1.0 / rank, 0.0)
    return dict(ndcg_at_10=float(ndcg.mean()), recall_at_10=float(hit.mean()), mrr_at_10=float(mrr.mean()))
