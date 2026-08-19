# it-shell
Asistente inteligente capaz de interpretar consultas formuladas en lenguaje natural y asistir al usuario en la exploración de herramientas CLI mediante técnicas modernas de Procesamiento del Lenguaje Natural (NLP), modelos de lenguaje basados en Transformers y recuperación semántica de información.

El sistema combina modelos de representación semántica (Sentence Transformers), bases de datos vectoriales y algoritmos de búsqueda por similitud para recuperar comandos relevantes, complementando los resultados con explicaciones, alternativas de uso, parámetros asociados y ejemplos de ejecución.

## dataset_builder
Constructor de la base de conocimientos.
Sus componentes:
- classifier: clasificador léxico de comando basado en descripciones.
- embedder: constructor de embeddings.
- indexer: constructor de indices.
- webcrawling: procesamiento de datos.

## semantic_search_engine
Encargado de procesar consulta de usuario, recuperar candidatos, construir una respuesta y retornarla.

## cli 

