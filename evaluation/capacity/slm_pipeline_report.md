# Evaluación exploratoria: pipeline de retrieval y asistente

**Fecha:** 2026-11-11 16:51:21
**Modelo configurado:** `qwen2.5:1.5b`
**Implementación medida:** `MockSLMAssistant`; no se invocó Qwen.
**Encoder de retrieval:** `sentence-transformers/all-mpnet-base-v2` (inglés)
**Total de consultas evaluadas:** 122 (86 desarrollo, 36 prueba)
**Categorías de evaluación:** `filesystem`, `networking`
**Category Boost configurado:** `+0.1`

> Los resultados de consultas en español no miden el desempeño del asistente: el encoder de retrieval es monolingüe inglés y el corpus está en inglés. No deben usarse para concluir sobre soporte multilingüe.
> En modo mock, la clasificación y selección son heurísticas locales; las métricas no son evidencia de precisión ni latencia de Qwen.

## 1. Comparativa de Estrategias de Retrieval (Hit@k y MRR)

| Estrategia de Retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Global (Sin Filtro)** | 36.1% | 47.5% | 59.8% | 65.6% | 0.4470 |
| **2. Clasificador simulado (heurísticas) Hard Filter (Exclusivo)** | 29.5% | 39.3% | 49.2% | 57.4% | 0.3721 |
| **3. Clasificador simulado (heurísticas) Soft Boost (+0.1)** | **38.5%** | **50.0%** | **58.2%** | **65.6%** | **0.4661** |
| **4. Oracle Hard Filter** | 29.5% | 43.4% | 48.4% | 54.9% | 0.3717 |
| **5. Oracle Soft Boost (+0.1)** | 36.9% | 49.2% | 55.7% | 63.1% | 0.4465 |

## 2. Rendimiento de Clasificación y Selección Final

| Componente | Métrica | Valor Obtenido | Descripción |
| :--- | :--- | :---: | :--- |
| **Clasificador simulado (heurísticas)** | Exactitud Categoría | **59.0%** | Predicción correcta entre filesystem y networking |
| **Clasificador simulado (heurísticas)** | Tasa Exclusión Errónea | **17.2%** | Casos donde una clasificación errónea afectaría un filtro hard |
| **Selección Final** | Baseline (Top-1 FAISS) | 38.5% | Elección directa del primer resultado |
| **Selección Final** | Selección simulada (heurísticas) | **32.0%** | Selección dentro de Top-K; comando ensamblado no equivale a validación funcional |

## 3. Rendimiento por Partición (Development vs Test)

| Partición | Consultas | Hit@1 (Global) | Hit@1 (Soft Boost) | Hit@5 (Soft Boost) | Selección Final (Baseline) | Selección mock |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development** | 86 | 38.4% | **43.0%** | **61.6%** | 43.0% | **37.2%** |
| **Test (Holdout)** | 36 | 30.6% | **27.8%** | **50.0%** | 27.8% | **19.4%** |

## 4. Conclusiones y Defensa Técnica

1. Estos resultados describen únicamente el modo mock y el encoder MPNet en inglés; no permiten afirmar que Qwen mejore clasificación, retrieval o selección.
2. La comparación español-inglés está confundida por el encoder monolingüe y el corpus en inglés; para evaluar el asistente multilingüe debe usarse un encoder/corpus compatible o limitar la métrica a consultas en inglés.
3. El contraste de category boost es exploratorio: requiere una comparación controlada y evidencia con categorías verificadas antes de recomendarlo como mejora general.
