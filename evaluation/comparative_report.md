# Comparative Retrieval Report — Phase 2

## Official evaluation

The official comparison uses 160 queries and the artifact snapshot recorded in
[`retrieval_snapshot_v2.json`](retrieval_snapshot_v2.json). FAISS and BM25 were
evaluated with the same corpus and query set, using a Top-15 candidate pool.

| Configuration | Snapshot comparable | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@5 | Method latency/query |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| FAISS + all-mpnet-base-v2 | Yes | 0.4125 | 0.6188 | 0.7125 | 0.8000 | 0.5377 | 0.3613 | 62.47 ms |
| BM25 Pyserini/Lucene | Yes | 0.2125 | 0.2625 | 0.2687 | 0.3125 | 0.2408 | 0.2056 | 3.99 ms |
| Hybrid BM25 + dense (α=0.5) | Yes | 0.3125 | 0.5375 | 0.6125 | 0.7312 | 0.4518 | 0.3073 | 172.76 ms |

BM25 is faster, but misses the phase acceptance thresholds (MRR > 0.25 and
Hit@5 >= 0.50). The hybrid substantially improves BM25, but remains below FAISS
on all quality metrics and has a higher measured total latency (229.73 ms/query
versus 62.47 ms/query for this FAISS run). Therefore FAISS remains the official
retrieval backend.

## Historical experiments

Previously generated reranker reports lack the snapshot checksum needed to
verify they used the same corpus and indices. They remain available for
reference, but are marked non-comparable by
[`comparative_table.md`](comparative_table.md) and are not used to select the
official configuration. Re-run them with the current snapshot before drawing
new conclusions.

## Reproducibility

The snapshot manifest records the corpus, query set, FAISS artifacts, and BM25
Lucene files by SHA-256. Detailed metrics and per-query results are in
[`comparative_all.json`](comparative_all.json) and
[`bm25_benchmark_v2_top15.json`](bm25_benchmark_v2_top15.json), with the hybrid
results in [`hybrid_benchmark_v2_top15.json`](hybrid_benchmark_v2_top15.json).
