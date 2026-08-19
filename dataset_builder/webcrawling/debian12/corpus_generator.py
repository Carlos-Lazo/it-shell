#!/usr/bin/env python3 
import re
import subprocess
import sys
import csv
 
# --- Configuracion ------------------------------------------------------
 
OUTPUT_FILE = "corpus.csv"
 
# Si es True, usa un shell interactivo (bash -ic) para que compgen -c
# tambien incluya alias y funciones definidas en ~/.bashrc.
# Si es False (default), usa bash -c, mas rapido y sin efectos secundarios.
INCLUIR_ALIAS_Y_FUNCIONES = False
 
# Indentacion maxima (en espacios) para que una linea que empieza con '-'
# se considere el TAG de una opcion (y no parte de su descripcion).
UMBRAL_INDENT_OPCION = 10
 
RE_ENCABEZADO = re.compile(r"^[A-Z][A-Z0-9 /\-]*$")
RE_BACKSPACE = re.compile(r".\x08")
 
# --------------------------------------------------------------------------
 
 
def obtener_comandos_compgen() -> list[str]:
    """Devuelve la lista ordenada y sin duplicados de `compgen -c`."""
    flag = "-ic" if INCLUIR_ALIAS_Y_FUNCIONES else "-c"
    try:
        resultado = subprocess.run(
            ["bash", flag, "compgen -c"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("bash no esta disponible en este sistema.")
 
    if resultado.returncode != 0:
        raise RuntimeError(f"compgen -c fallo: {resultado.stderr.strip()}")
 
    comandos = {linea.strip() for linea in resultado.stdout.splitlines() if linea.strip()}
    return sorted(comandos)
 
 
def obtener_texto_man(comando: str) -> str | None:
    """Ejecuta `man <comando>`  y devuelve el texto plano.
 
    Devuelve None si el comando no tiene pagina de manual.
    """
    env = {
        "LC_ALL": "C",
        "LANG": "C",
        "LANGUAGE": "en",
        "MANWIDTH": "80",
        "MANPAGER": "cat",
        "PAGER": "cat",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    try:
        resultado = subprocess.run(
            ["man", comando],
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        raise RuntimeError("El comando 'man' no esta disponible en este sistema.")
 
    if resultado.returncode != 0 or not resultado.stdout.strip():
        return None
 
    return RE_BACKSPACE.sub("", resultado.stdout)
 
 
def parsear_secciones(texto: str) -> dict[str, list[str]]:
    """Agrupa las lineas del texto de man por seccion (NAME, OPTIONS, etc.)."""
    secciones: dict[str, list[str]] = {}
    seccion_actual = None
 
    for linea in texto.split("\n"):
        if linea and not linea.startswith((" ", "\t")) and RE_ENCABEZADO.match(linea.strip()):
            seccion_actual = linea.strip()
            secciones[seccion_actual] = []
        elif seccion_actual is not None:
            secciones[seccion_actual].append(linea)
 
    return secciones
 
 
def extraer_descripcion_base(lineas_name: list[str]) -> str:
    """Extrae la descripcion de la seccion NAME ('comando - descripcion')."""
    texto_name = " ".join(l.strip() for l in lineas_name if l.strip())
 
    if " - " in texto_name:
        return texto_name.split(" - ", 1)[1].strip()
 
    return texto_name
 
 
def extraer_sinopsis(lineas: list[str]) -> str:
    """Extrae el texto de la seccion SYNOPSIS, colapsado en una sola linea."""
    texto = " ".join(l.strip() for l in lineas if l.strip())
    return re.sub(r"\s+", " ", texto).strip()
 
 
def extraer_opciones(lineas: list[str]) -> list[tuple[str, str]]:
    """Extrae pares (opcion, descripcion) de una lista de lineas de man."""
    opciones: list[tuple[str, str]] = []
    opcion_actual = None
    desc_actual: list[str] = []
 
    def guardar_actual():
        if opcion_actual is not None:
            opciones.append((opcion_actual, " ".join(desc_actual).strip()))
 
    for linea in lineas:
        if not linea.strip():
            continue
 
        indent = len(linea) - len(linea.lstrip(" "))
        contenido = linea.strip()
 
        if contenido.startswith("-") and indent <= UMBRAL_INDENT_OPCION:
            guardar_actual()
            opcion_actual = contenido
            desc_actual = []
        elif opcion_actual is not None:
            desc_actual.append(contenido)
 
    guardar_actual()
    return opciones
 
 
def procesar_comando(
    comando: str,
) -> tuple[str, str, list[tuple[str, str]]] | None:
    """Devuelve (descripcion_base, sinopsis, [(opcion, descripcion), ...]) o None."""
    texto = obtener_texto_man(comando)
    if texto is None:
        return None
 
    secciones = parsear_secciones(texto)
    descripcion_base = extraer_descripcion_base(secciones.get("NAME", []))
    sinopsis = extraer_sinopsis(secciones.get("SYNOPSIS", []))
    lineas_opciones = secciones.get("OPTIONS") or secciones.get("DESCRIPTION") or []
    opciones = extraer_opciones(lineas_opciones)
 
    return descripcion_base, sinopsis, opciones
 
 
def generar_csv(output_path: str) -> tuple[list[str], list[str]]:
    """Procesa cada comando de compgen -c y escribe el CSV de salida.
 
    Retorna (comandos_procesados, comandos_sin_datos).
    """
    comandos = obtener_comandos_compgen()
    procesados = []
    sin_datos = []
 
    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
 
        for comando in comandos:
            resultado = procesar_comando(comando)
 
            if resultado is None:
                sin_datos.append(comando)
                continue
 
            descripcion_base, sinopsis, opciones = resultado
            writer.writerow([comando, "", descripcion_base])
            if sinopsis:
                writer.writerow([comando, "SYNOPSIS", sinopsis])
            for opcion, desc_opcion in opciones:
                writer.writerow([comando, opcion, desc_opcion])
 
            procesados.append(comando)
 
    return procesados, sin_datos
 
 
if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_FILE
 
    procesados, sin_datos = generar_csv(out_path)
 
    print(f"Listo. Archivo generado: {out_path}")
    print(f"Total de comandos encontrados por compgen -c: {len(procesados) + len(sin_datos)}")
    print(f"Comandos procesados con exito ({len(procesados)}).")
    print(f"Comandos SIN datos de man ({len(sin_datos)}).")
 
