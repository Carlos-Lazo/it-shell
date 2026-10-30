# Integración MCP de it-shell

it-shell expone sus operaciones mediante el SDK oficial MCP para Python y transporte
`stdio`. El servidor ofrece herramientas de consulta y validación; **ninguna herramienta
ejecuta comandos de shell**.

## Requisitos

1. Python 3.10 o superior.
2. Dependencias del motor y el SDK MCP:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r semantic_search_engine/requirements.txt
   ```

3. Corpus YAML y artefactos configurados en
   `semantic_search_engine/config/default.yaml`: índice FAISS, aliases y modelo
   `sentence-transformers/all-mpnet-base-v2`. El corpus y algunos artefactos generados
   no se distribuyen en Git. Consulta el [README principal](../README.md) para prepararlos.

## Iniciar el servidor

Desde la raíz del repositorio:

```bash
PYTHONPATH=. .venv/bin/python -m semantic_search_engine.mcp.server
```

Para usar otro archivo de configuración:

```bash
PYTHONPATH=. .venv/bin/python -m semantic_search_engine.mcp.server \
  --config semantic_search_engine/config/default.yaml
```

El proceso usa `stdio` para el protocolo MCP. No redirijas ni escribas mensajes propios
en stdout mientras está conectado a un cliente; stdout queda reservado para JSON-RPC.

## Herramientas disponibles

### `search_commands`

- `query` (texto requerido): consulta en lenguaje natural, hasta 2.000 caracteres.
- `top_k` (entero opcional): candidatos por paso; entre 1 y 50, valor inicial 10.
- Devuelve los candidatos con alias, intención, comando, ejemplo y puntuaciones.

### `plan_operation`

- `query` (texto requerido): consulta en lenguaje natural, hasta 2.000 caracteres.
- `top_k` (entero opcional): entre 1 y 50, valor inicial 10.
- Devuelve análisis, pasos, dependencias, propuesta, avisos y explicación estructurados.

### `validate_command`

- `command` (texto requerido): comando para inspección, hasta 4.096 caracteres.
- Devuelve el resultado del `ProposalValidator`, con severidad, códigos, mensajes y
  recomendaciones. El texto se analiza; nunca se envía a una shell.

Los argumentos vacíos, demasiado largos o fuera de rango se rechazan con un error
explícito. Los problemas al cargar corpus, índices o modelo también se propagan como
errores, no como una respuesta exitosa vacía.

## Configuración de ejemplo para Claude Desktop

Reemplaza las rutas por las absolutas de tu equipo. El directorio de trabajo debe ser la
raíz del repositorio:

```json
{
  "mcpServers": {
    "it-shell": {
      "command": "/ruta/al/repositorio/.venv/bin/python",
      "args": [
        "-m",
        "semantic_search_engine.mcp.server",
        "--config",
        "semantic_search_engine/config/default.yaml"
      ],
      "cwd": "/ruta/al/repositorio",
      "env": {
        "PYTHONPATH": "/ruta/al/repositorio"
      }
    }
  }
}
```

Reinicia el cliente después de guardar su configuración y confirma que aparecen las tres
herramientas.

## Pruebas y latencia

Las pruebas unitarias usan motores falsos y no requieren índice ni modelo:

```bash
PYTHONPATH=. python -m pytest tests/test_mcp_tools.py -q
```

Medición de referencia local, con CPU, artefactos del repositorio y `top_k=10`: en diez
solicitudes calientes, `plan_operation` obtuvo 162,24 ms de mediana y 341,52 ms de p95;
`search_commands`, 152,83 ms de mediana y 162,02 ms de p95. La primera búsqueda tardó
15,91 s al cargar el modelo MPNet. Son resultados de una única máquina y no garantizan
la misma latencia en otro entorno. La fase queda cerrada con esta validación local; si
se ejecuta en otro equipo, medir allí con el proceso iniciado y el modelo cargado, y
reportar por separado el arranque frío y la primera carga.
