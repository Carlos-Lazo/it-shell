| Config | Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@1 | nDCG@5 | Total latency ms |
|--------|-------|------:|------:|------:|-------:|-----:|-------:|-------:|-----------------:|
| Baseline (FAISS MPNet) | FAISS + MPNet | 0.4062 | 0.5813 | 0.6562 | 0.7875 | 0.5211 | 0.4062 | 0.3494 | 1055.92 |
| Reranker DistilRoBERTa‑base | stsb‑distilroberta‑base | 0.4125 | 0.5938 | 0.6375 | 0.7937 | 0.5291 | 0.4125 | 0.3440 | 3021.09 |
| Reranker BGE‑base | bge‑reranker‑base | 0.3625 | 0.5312 | 0.6562 | 0.7937 | 0.4861 | 0.3625 | 0.3215 | 6044.90 |
