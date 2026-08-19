## Classifier

Clasificador léxico basado en algoritmo tf-idf. 

### Preparación

Con base en un archivo de corpus de la forma 'comando, descripcíon, categoria', asignando pesos a los términos frecuentes involucrados en las descripciones de cada comando con respecto a una categoría. El resultado de este proceso es `vocabulary.json`.
El corpus base del entrenamiento del vocabulario es 'data/commands.csv', el vocabulario final tiene esta forma:

Filesystem
| palabra    | TF-IDF |
| ---------- | -----: |
| directory  |   0.91 |
| inode      |   1.00 |
| filesystem |   0.99 |
| file       |   0.42 |
| path       |   0.81 |

Networking
| palabra | TF-IDF |
| ------- | -----: |
| icmp    |   1.00 |
| packet  |   0.89 |
| socket  |   0.84 |
| dns     |   0.93 |

...

<br>

__Para la generación del vocabulario:__
``go run cmd/main.go --train``

<br>



### Run
Para consultar la clasificación posible de un nuevo comando se debe ejecutar:
`go run cmd/main.go <descripcion_comando>`, utiliza el documento de la sección anterior. <br>
En caso de modificación en el corpus base de clasificación, se puede invocar la flag ``-train`` para vovler a calcular los pesos. Por defecto es `false`.

## Generate
El programa del directorio `generatecorpus/` genera los archivos corpus/*.yaml; clasificando los comando antes generados de `webclawling/debian12/corpus.csv` utilizando el clasificador de la seccion _Preparacion_. Cada archivo .yaml contiene los documentos para generar la representación vectorial. Esta organización será util para posterior indexación.
### Ejemplo
Los documentos tienen esta forma: _filesystem.yaml_
```bash
category: "filesystem"
documents:
- id: 120
  intent: "Include  dotfiles  when  compiling  and  copying  non-compilable files."
  command: "babeljs-7"
  example: "babeljs-7 --include-dotfiles"
- id: 121
  intent: "Exclude ignored files when copying non-compilable files."
  command: "babeljs-7"
  example: "babeljs-7 --no-copy-ignored"
  ...

  ```
