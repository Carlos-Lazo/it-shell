# Comparación BGE-M3: reglas vs. Ollama

- Consultas: 122 (86 development, 36 test).
- Índice: 27413 vectores BGE-M3, 1024 dimensiones.
- Ollama: `0.40.13`, `qwen2.5:1.5b`, digest `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`.
- Ollama candidate-selection calls: 122; selection fallbacks: 1.
- El fallback ocurrió en `networking.tar::3` (development/en): se conservó `mksquashfs`, que no coincide con la etiqueta `tar`, y cuenta como selección incorrecta. El informe no registra la causa de la excepción.
- El test se reporta con parámetros fijos; no se usó para seleccionar configuración.
- La métrica de selección evalúa el comando del alias elegido; no valida todos sus argumentos ni ejecuta comandos.

## Resultado general

| Método | BGE Hit@1 | BGE Hit@5 | BGE Hit@10 | BGE MRR | Selección correcta | Latencia media |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.393 | 0.680 | 0.787 | 0.523 | 0.393 | 7.22 s |
| ollama | 0.393 | 0.680 | 0.787 | 0.523 | 0.549 | 20.45 s |

## Por partición e idioma

| Partición/idioma | Método | Consultas | Hit@10 | MRR | Selección final correcta |
|---|---|---:|---:|---:|---:|
| development/en | rules | 57 | 0.825 | 0.574 | 0.456 |
| development/en | ollama | 57 | 0.825 | 0.574 | 0.614 |
| development/es | rules | 29 | 0.724 | 0.549 | 0.414 |
| development/es | ollama | 29 | 0.724 | 0.549 | 0.552 |
| test/en | rules | 24 | 0.750 | 0.468 | 0.333 |
| test/en | ollama | 24 | 0.750 | 0.468 | 0.500 |
| test/es | rules | 12 | 0.833 | 0.335 | 0.167 |
| test/es | ollama | 12 | 0.833 | 0.335 | 0.333 |

## Procedencia y límites

Esta primera comparación aísla la selección final: ambos métodos reciben el mismo ranking BGE-M3 para la consulta original. Rules selecciona el primer candidato que supera el umbral; Ollama selecciona entre ese mismo conjunto de hasta 5 candidatos elegibles. No se evalúan reformulación de consultas, análisis de pasos ni calidad de explicación. El comando ensamblado por Ollama se ignora; la métrica cuenta únicamente si el comando documentado del alias seleccionado coincide con una etiqueta relevante.

The BGE-M3 aliases align to the current indexed corpus subset, but the BGE vector artifact does not include hashes of source text. Exact source-text revision cannot be independently verified.

El retrieval es compartido, por lo que Hit@k y MRR describen BGE-M3 y son idénticos para ambas condiciones. La comparación del selector usa selected_command_correct. El benchmark mide acierto del comando seleccionado, no ejecución funcional ni corrección integral de argumentos. Las métricas por idioma describen esta muestra pequeña de capacidades.
