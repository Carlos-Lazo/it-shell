## Retrieval service
procesa la consulta y recupera candidatos por medio de alias.

#### Componentes
- ``query_analyzer``: normaliza la consulta, identifica idioma, intención y complejidad.
- ``planner``: descompone consultas complejas en subconsultas ordenadas.
- ``:retriever``: consulta FAISS y recupera candidatos usando los aliases ``categori:id``.
- ``service``: punto de entrada interno del servicio.

### Ejemplo
<p>La consulta:<p>

``buscar logs grandes, comprimirlos y enviarlos al servidor``

<p>Se convierte en:<p>

```
1. buscar archivos log grandes
2. comprimir archivos
3. transferir archivo a servidor
```