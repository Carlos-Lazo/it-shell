# Dataset builder
Componente off-line del proyecto, encargado de la extracción, transformación y normalización de los datos.


## Flujo de generacion de datos

1) webcrawling/
    1) `debian12/corpus_generator.py` -> /corpus.csv
    2) `linux/linux_ultimate_commands_list.py` -> /linux_ultimate_commands_list.csv
    3) `git/git_commands_list.csv`
    4) `docker/docker_crawler.py` -> /docker_cli_commands_list.csv

2) classifier
    1) `cmd/main.go --train` -> vocabulary.json 
    2) `cmd/generatecorpus/main.go` -> corpus/*.yaml
    3) `cmd/generategit/main.go` -> corpus/git.yaml
    4) `cmd/generatedocker/main.go` -> corpus/docker.yaml



