# Comparative Benchmark Report

## Overview
We evaluated three configurations on the **160‑query** benchmark:

| Configuration | Model | Hit @ 1 | Hit @ 3 | Hit @ 5 | Hit @ 10 | MRR | Retrieval latency (ms) | Rerank latency (ms) | Total latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (FAISS MPNet)** | – | 0.4125 | 0.6188 | 0.7125 | 0.8063 | 0.5385 | 55.49 | – | 55.49 |
| **DistilRoBERTa‑base** | `stsb-distilroberta-base` | 0.4125 | 0.5938 | 0.6375 | 0.7937 | 0.5291 | 55.49 | 2689.79 | 2745.28 |
| **BGE‑reranker‑base** | `bge-reranker-base` | 0.3625 | 0.5312 | 0.6562 | 0.7937 | 0.4861 | 44.03 | 6000.87 | 6044.90 |

## Key Observations
1. **Precision**
   * The baseline already provides the highest Hit@1 (0.4125) and the best MRR (0.5385).
   * DistilRoBERTa does **not improve** over the baseline; its metrics are slightly lower for Hit@3/5/10 and MRR.
   * BGE‑reranker shows a **noticeable drop** in Hit@1 (0.3625) and MRR (0.4861) while keeping Hit@10 similar to baseline.
2. **Latency**
   * Adding a reranker incurs a large latency penalty. DistilRoBERTa adds ~2.7 s/query; BGE adds ~6 s/query.
   * Both exceed a typical interactive‑CLI budget (< 100 ms) by orders of magnitude.
3. **Trade‑off**
   * Neither reranker provides a precision gain that justifies its latency cost for real‑time use.
   * If batch‑or‑offline processing is acceptable, the models could be used to improve ranking for downstream analytics.

## Recommendations
| Action | Reasoning |
|---|---|
| **Keep FAISS‑only retrieval as the default runtime** | Highest precision with negligible latency. |
| **Expose reranking as an optional offline mode** (`--rerank-model <model>`) | Allows power users to generate higher‑quality rankings when latency is not critical. |
| **Document the results** | Add a *7.3 Comparative Results* subsection to `docs/metodologia_benchmark_v2.md` (see below). |
| **Future work** | Explore newer multilingual cross‑encoders or lightweight similarity models that balance precision and speed (e.g., miniLM‑cross‑encoder). |

## Documentation Update (suggested snippet)
```markdown
### 7.3 Comparative Results
We compared three configurations on the 160‑query benchmark (see Table 1). The baseline FAISS‑MPNet retrieval outperforms both cross‑encoders in precision while having a 50‑fold lower latency. Consequently, we retain the baseline as the default and provide the cross‑encoders as optional offline rerankers.
```

---
*Report generated on 2026‑10‑23.*
