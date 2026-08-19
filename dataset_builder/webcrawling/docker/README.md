# Extracción de comandos docker

Se utiliza la web [CLI docker reference]("https://docs.docker.com/reference/cli/docker/") como fuente.


### Extracción

Utiliza `re, httpx y BeautifulSoup` para generar un cliente y parsear el código HTML de la web objetivo, obteniendo:
  1) Nombre del comando
  2) Descripcion funcional breve
  3) Enlace de origen.
 
Combina los items anteriores para generar un CSV con columnas: comando, descripción y enlace.
La fila de descripcion del comando base tiene 'Sin descripción disponible'.

### Run
```bash
python docker_crawler.py
```
(OUTPUT_FILE por defecto "docker_cli_comandos_list.csv")

### Ejemplo:
- docker build,Sin descripción disponible,https://docs.docker.com/reference/cli/docker/buildx/build/
- docker build --add-host,Add a custom host-to-IP mapping (format:host:ip),https://docs.docker.com/reference/cli/docker/buildx/build/
- docker build --allow,"Allow extra privileged entitlement (e.g.,network.host,security.insecure,device,buildx.local.delete)",https://docs.docker.com/reference/cli/docker/

