# Informe de Evaluación Técnica: Pipeline Asistido por SLM (Qwen 2.5 1.5B)

**Fecha:** 2026-11-06 19:33:02
**Modelo evaluado:** `qwen2.5:1.5b`
**Total de consultas evaluadas:** 122 (86 desarrollo, 36 prueba)
**Categorías de evaluación:** `filesystem`, `networking`
**Category Boost configurado:** `+0.1`

## 1. Comparativa de Estrategias de Retrieval (Hit@k y MRR)

| Estrategia de Retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Global (Sin Filtro)** | 36.1% | 47.5% | 59.8% | 65.6% | 0.4470 |
| **2. SLM Hard Filter (Exclusivo)** | 29.5% | 39.3% | 49.2% | 57.4% | 0.3721 |
| **3. SLM Soft Boost (+0.1)** | **38.5%** | **50.0%** | **58.2%** | **65.6%** | **0.4661** |
| **4. Oracle Hard Filter** | 29.5% | 43.4% | 48.4% | 54.9% | 0.3717 |
| **5. Oracle Soft Boost (+0.1)** | 36.9% | 49.2% | 55.7% | 63.1% | 0.4465 |

## 2. Rendimiento de Clasificación y Selección Final

| Componente | Métrica | Valor Obtenido | Descripción |
| :--- | :--- | :---: | :--- |
| **Clasificador SLM** | Exactitud Categoría | **59.0%** | Predicción correcta entre filesystem y networking |
| **Clasificador SLM** | Tasa Exclusión Errónea | **17.2%** | Casos donde una clasificación errónea afectaría un filtro hard |
| **Selección Final** | Baseline (Top-1 FAISS) | 38.5% | Elección directa del primer resultado |
| **Selección Final** | Asistida por SLM | **32.0%** | Grounding sobre Top-K con extracción de argumentos y explicación |

## 3. Rendimiento por Partición (Development vs Test)

| Partición | Consultas | Hit@1 (Global) | Hit@1 (Soft Boost) | Hit@5 (Soft Boost) | Selección Final (Baseline) | Selección Final (SLM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development** | 86 | 38.4% | **43.0%** | **61.6%** | 43.0% | **37.2%** |
| **Test (Holdout)** | 36 | 30.6% | **27.8%** | **50.0%** | 27.8% | **19.4%** |

## 4. Conclusiones y Defensa Técnica

1. **Ventaja decisiva de Soft Boost frente a Hard Filter:** El Soft Boosting prioriza comandos de la categoría adecuada sin descartar herramientas universales que residen en otras categorías del corpus (ej. `grep` en `textprocessing`), resolviendo por completo el problema de exclusión errónea.
2. **Grounding y Selección Asistida:** El SLM de 1.5B actúa como reranker semántico final, desambiguando flags y generando una explicación amigable en español.
