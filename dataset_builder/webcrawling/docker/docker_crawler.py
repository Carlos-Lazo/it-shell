import re
import httpx
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://docs.docker.com/reference/cli/docker/"

# Usamos un cliente HTTP eficiente con cabecera de User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

OUTPUT_FILE_NAME = "docker_cli_comandos_list.csv"

with httpx.Client(headers=headers, follow_redirects=True, timeout=15.0) as client:
    print("1. Cargando la página principal para extraer enlaces...")
    res = client.get(base_url)
    
    soup = BeautifulSoup(res.text, "html.parser")
    sidebar = soup.select_one("aside nav, nav[class*='navbar-font'], [class*='navbar-font'] nav")

    commands_list = []
    if sidebar:
        links = sidebar.find_all("a")
        seen = set()

        for link in links:
            text = link.get_text(strip=True)
            href = link.get("href", "")
            
            if href and "#" not in href and "/cli/" in href and text not in seen:
                seen.add(text)
                full_url = href if href.startswith("http") else f"https://docs.docker.com{href}"
                commands_list.append({"comando_base": text, "url": full_url})
    
    print(f"Se encontraron {len(commands_list)} comandos. Extrayendo descripciones y opciones...\n")

    full_dataset = []

    # 2. Iteramos por cada comando con peticiones HTTP livianas
    for item in commands_list:
        cmd_base = item["comando_base"]
        cmd_url = item["url"]
        
        print(f"Procesando: {cmd_base}")
        
        try:
            cmd_res = client.get(cmd_url)
            if cmd_res.status_code != 200:
                print(f"  -> No se pudo obtener {cmd_url} (HTTP {cmd_res.status_code})")
                continue
                
            cmd_soup = BeautifulSoup(cmd_res.text, "html.parser")
            
            cmd_base_clean = re.sub(r'["]', '', cmd_base).strip()

            # --- EXTRAER DESCRIPCIÓN DEL COMANDO ---
            desc_text = ""
            desc_header = cmd_soup.find(id="description")
            if desc_header:
                sibling = desc_header.find_next_sibling()
                while sibling and sibling.name in ["a", "blockquote"]:
                    desc_text += sibling.get_text(strip=True) + " "
                    sibling = sibling.find_next_sibling()
            
            desc_text.replace('"', '').strip()

            # PASO A: Insertar el comando original
            full_dataset.append({
                "Comando": cmd_base_clean,
                "Descripción": desc_text if desc_text else "Sin descripción disponible",
                "Enlace": cmd_url
            })

            # PASO B: Buscar opciones
            options_header = cmd_soup.find(id="options")
            if options_header:
                options_table = options_header.find_next("table")
                if options_table:
                    rows = options_table.find_all("tr")
                    for row in rows[1:]:  # Omitir cabecera
                        cols = row.find_all(["td", "th"])
                        if len(cols) >= 3:
                            option_flag = cols[0].get_text(strip=True)
                            option_desc = cols[2].get_text(strip=True)
                            option_desc.replace('"', '')
                            full_dataset.append({
                                "Comando": f"{cmd_base_clean} {option_flag}",
                                "Descripción": option_desc,
                                "Enlace": cmd_url
                            })

        except Exception as e:
            print(f"Error procesando {cmd_url}: {e}")

# 3. Guardar en CSV
df = pd.DataFrame(full_dataset)
print(f"\n Se procesaron {len(df)} comandos válidos.")

csv_filename = OUTPUT_FILE_NAME
df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
print(f"\n Se ha generado '{csv_filename}' exitosamente.")