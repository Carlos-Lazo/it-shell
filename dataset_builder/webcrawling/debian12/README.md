### Extracción

Obtiene la lista de comandos disponibles con `compgen -c`, ejecuta `man <comando>` para cada uno y extrae:
  1) la descripcion base (seccion NAME: "comando - descripcion")
  2) las opciones y sus descripciones (seccion OPTIONS, o DESCRIPTION si
     OPTIONS no existe), identificando cada opcion por lineas que empiezan
     con '-' y tienen poca indentacion.
 
Genera un CSV con columnas: comando, opcion, descripcion.
La fila de descripcion base tiene 'opcion' vacio. Los comandos sin pagina
de manual (builtins, keywords, alias, etc.) se listan aparte al final
y no se escriben en el CSV.

### Run
```bash
python corpus_generator.py
```

### Ejemplo:
- ls,,list directory contents
- ls,SYNOPSIS,ls [OPTION]... [FILE]...
- ls,-a --all,do not ignore entries starting with .
- ...

### Uso:
```bash
    python corpus_generator.py corpus0.csv
```
(sin argumentos usa OUTPUT_FILE por defecto)