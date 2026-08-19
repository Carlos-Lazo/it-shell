import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "https://www.linux.org/threads/the-ultimate-list-of-linux-terminal-commands.57914/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

OUTPUT_FILE = "linux_ultimate_commands_list.csv"
print("Obteniendo la página web...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    code_blocks = soup.select(".bbCodeBlock-content, div.bbCodeBlock-content")

    if code_blocks:
        selected_blocks = code_blocks[:2]
        print(f"Se encontraron {len(code_blocks)} bloques. Procesando los primeros {len(selected_blocks)}...")

        data = []
        current_category = "Sin Categoría"
        
        
        for index, block in enumerate(selected_blocks, start=1):
            text_content = block.get_text()
            lines = text_content.splitlines()

            for line in lines:
                line_str = line.strip()

                # detectar nueva categoría
                if line_str.startswith("##"):
                    current_category = line_str.replace("##", "").strip()
                    current_category = line_str.replace("#", "").strip()
                    current_category = re.sub(r'\s*\(\d+\)', '', current_category)
                # detectar líneas de comandos
                elif line_str.startswith("-"):
                    content = line_str.lstrip("-").strip()

                    # separar el comando bruto de texto extra
                    parts = re.split(r",", content, maxsplit=1)
                    raw_cmd = parts[0].strip()
                    
                    
                    # si contiene '?' o comienza con '[' se ignora el comando completo
                    if '?' in raw_cmd or  raw_cmd.startswith('[') or 'Undocumented' in current_category:
                        continue


                    # unicamente secuencias alfanuméricas (ignoramos símbolos como ++, *, ***)
                    words_only = re.findall(r'[a-zA-Z0-9]+', raw_cmd)
                    if len(words_only) > 1:
                        continue
                    
                    if '***' in raw_cmd:
                        origen = "non-free component"
                    elif '**' in raw_cmd:
                        origen = "contrib"
                    elif '*' in raw_cmd:
                        origen = "not by default"
                    elif '+' in raw_cmd:  # cubre tanto '+' como '++'
                        origen = "bash sell builtins"
                    else:
                        origen = "unknown"  # valor por defecto cuando no tiene ninguno de los símbolos anteriores

                    
                    # elimina paréntesis y su contenido
                    cmd_no_parentheses = re.sub(r'\(.*?\)', '', raw_cmd)
                    # elimina símbolos *, +, (, ), ' y "
                    comando = re.sub(r'[*+`()\'"]', '', cmd_no_parentheses).strip()

    
                    data.append({                
                        "Comando": comando,
                        "Categoría": current_category,
                        "Origen": origen,
                    })

        
        # guardar en csv
        df = pd.DataFrame(data)
        csv_filename = OUTPUT_FILE
        df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

        print(f"\nSe procesaron {len(df)} comandos válidos.")
        
        print(f"Archivo guardado como: '{csv_filename}'")
    else:
        print("No se encontraron bloques bbCodeBlock-content en la página.")
else:
    print(f"Error al acceder a la página. Código: {response.status_code}")