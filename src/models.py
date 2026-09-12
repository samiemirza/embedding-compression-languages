"""Small multilingual embedding models (<600M params) and their prompt conventions."""
MODELS = {
    # key: (hf_id, query_prefix, doc_prefix, trust_remote_code[, model_kwargs])
    "me5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: ", False),
    "me5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: ", False),
    "me5-large": ("intfloat/multilingual-e5-large", "query: ", "passage: ", False),
    "bge-m3": ("BAAI/bge-m3", "", "", False),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B",
                   "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", False,
                   {"torch_dtype": "float16"}),
}
