import argparse
import json
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


DEFAULT_CORPUS = Path("dataset_builder/corpus/ollama_corpus")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "corpus_paraphrases"
SYSTEM_PROMPT = """You generate search-query paraphrases from a corpus record's
intent field. The user's message contains only the intent text.

Return only a JSON object with this shape:
{"phrases": ["...", "..."]}

Rewrite the intent as 1 to 4 short, natural search queries a user could type
to find that capability. The phrases must be semantic paraphrases of the intent,
not commands, examples, or option values. Keep the same language as the intent
and preserve its meaning. Do not add capabilities or details that the intent
does not state. Return {"phrases": []} only if the intent is empty or has no
understandable meaning. Never output executable shell commands."""
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "phrases": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1},
        }
    },
    "required": ["phrases"],
    "additionalProperties": False,
}


def iter_corpus_files(corpus_path: Path) -> Iterator[Path]:
    if corpus_path.is_file():
        files = [corpus_path]
    elif corpus_path.is_dir():
        files = sorted((*corpus_path.glob("*.yaml"), *corpus_path.glob("*.yml")))
    else:
        raise FileNotFoundError(f"No se encontró el corpus: {corpus_path}")

    if not files:
        raise ValueError(f"No hay archivos YAML en el corpus: {corpus_path}")
    yield from files


def load_records(corpus_file: Path) -> list[dict[str, str]]:
    try:
        data = yaml.safe_load(corpus_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"YAML inválido en {corpus_file}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        raise ValueError(f"Formato de corpus inválido en {corpus_file}: falta documents")

    category = data.get("category", corpus_file.stem)
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"Categoría inválida en {corpus_file}")

    records = []
    aliases = set()
    for row in data["documents"]:
        if not isinstance(row, dict) or row.get("id") is None:
            raise ValueError(f"Registro inválido o sin id en {corpus_file}")
        alias = f"{category}:{row['id']}"
        if alias in aliases:
            raise ValueError(f"Alias duplicado en {corpus_file}: {alias}")
        aliases.add(alias)
        records.append({
            "alias": alias,
            "intent": str(row.get("intent") or ""),
        })
    return records


def parse_phrases(content: str, alias: str) -> list[str]:
    try:
        response = json.loads(content)
    except json.JSONDecodeError as error:
        preview = content[:300].replace("\n", " ")
        raise ValueError(
            f"Ollama devolvió JSON inválido para {alias}: {preview!r}"
        ) from error

    if not isinstance(response, dict) or set(response) != {"phrases"}:
        raise ValueError(
            f"Respuesta inválida para {alias}: se esperaba un objeto con la clave phrases"
        )

    phrases = response["phrases"]
    if not isinstance(phrases, list):
        raise ValueError(f"Respuesta inválida para {alias}: phrases no es una lista")
    if len(phrases) > 4:
        raise ValueError(
            f"Respuesta inválida para {alias}: devolvió {len(phrases)} frases; máximo 4"
        )
    if not all(isinstance(phrase, str) and phrase.strip() for phrase in phrases):
        raise ValueError(
            f"Respuesta inválida para {alias}: todas las frases deben ser textos no vacíos"
        )

    return [phrase.strip() for phrase in phrases]


def generate_phrases(
    model: str,
    ollama_url: str,
    record: dict[str, str],
    timeout: int,
    retries: int = 2,
) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps({"intent": record["intent"]}, ensure_ascii=False),
        },
    ]
    last_error = None

    for attempt in range(retries + 1):
        payload = {
            "model": model,
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 2048},
            "messages": messages,
        }
        request = Request(
            f"{ollama_url.rstrip('/')}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                result: Any = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama respondió HTTP {error.code}: {detail}") from error
        except (TimeoutError, URLError) as error:
            if attempt < retries:
                delay = min(2 ** attempt, 10)
                print(
                    f"[WARN] {record['alias']}: timeout/error de conexión "
                    f"(intento {attempt + 1}/{retries + 1}); reintento en {delay}s"
                )
                time.sleep(delay)
                continue
            reason = error.reason if isinstance(error, URLError) else error
            raise RuntimeError(
                f"Fallo de conexión con Ollama para {record['alias']} tras "
                f"{retries + 1} intentos: {reason}"
            ) from error

        if not isinstance(result, dict):
            last_error = ValueError(f"Respuesta inválida de Ollama para {record['alias']}")
        else:
            message = result.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                last_error = ValueError(
                    f"Ollama no devolvió contenido para {record['alias']}"
                )
            else:
                content = message["content"]
                try:
                    return parse_phrases(content, record["alias"])
                except ValueError as error:
                    last_error = error
                    if attempt < retries:
                        messages.extend([
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "La respuesta anterior no cumplió el formato requerido. "
                                    "Inténtalo de nuevo: responde solo con un objeto JSON "
                                    'con la clave "phrases" y hasta cuatro textos no vacíos.'
                                ),
                            },
                        ])
                        continue

        if attempt < retries:
            messages.extend([
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": (
                        'Responde solo con un objeto JSON válido con la clave "phrases" '
                        "y hasta cuatro textos no vacíos."
                    ),
                },
            ])

    raise ValueError(
        f"Respuesta inválida para {record['alias']} tras {retries + 1} intentos: "
        f"{last_error}"
    ) from last_error


def load_existing_results(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(alias, str)
        and isinstance(phrases, list)
        and all(isinstance(phrase, str) for phrase in phrases)
        for alias, phrases in data.items()
    ):
        raise ValueError(f"El archivo de salida no tiene el formato esperado: {path}")
    return data


def load_existing_errors(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(alias, str) and isinstance(error, str)
        for alias, error in data.items()
    ):
        raise ValueError(f"El archivo de errores no tiene el formato esperado: {path}")
    return data


def save_results(path: Path, results: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(results, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera paráfrasis de búsqueda para registros YAML usando Ollama."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help=f"Archivo YAML o directorio de corpus (default: {DEFAULT_CORPUS})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directorio para JSON por categoría (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--model", default="qwen2.5:3b",
                        help="Modelo local de Ollama (default: qwen2.5:3b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="URL base de la API de Ollama")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout por registro en segundos (default: 300)")
    parser.add_argument("--retries", type=int, default=2,
                        help="Reintentos ante respuestas inválidas (default: 2)")
    parser.add_argument("--limit", type=int,
                        help="Cantidad máxima de registros a generar en esta ejecución")
    parser.add_argument("--resume", action="store_true",
                        help="Continúa desde aliases ya guardados en el archivo de salida")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero")
    if args.retries < 0:
        parser.error("--retries no puede ser negativo")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser mayor que cero")

    corpus_files = list(iter_corpus_files(args.corpus))
    output_paths = [args.output_dir / f"{path.stem}.json" for path in corpus_files]
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.resume:
        parser.error(
            f"Ya existen archivos de salida ({existing_outputs[0]}); "
            "usa --resume o elige otro --output-dir"
        )

    remaining_limit = args.limit
    work = []
    for corpus_file, output_path in zip(corpus_files, output_paths):
        records = load_records(corpus_file)
        results = load_existing_results(output_path) if args.resume else {}
        pending = [record for record in records if record["alias"] not in results]
        if remaining_limit is not None:
            pending = pending[:remaining_limit]
            remaining_limit -= len(pending)
        errors_path = output_path.with_name(f"{output_path.stem}.errors.json")
        errors = load_existing_errors(errors_path) if args.resume else {}
        work.append((corpus_file, output_path, errors_path, results, errors, pending))

    total_pending = sum(len(pending) for _, _, _, _, _, pending in work)
    print(f"Registros pendientes: {total_pending}; modelo: {args.model}")
    for corpus_file, output_path, errors_path, results, errors, pending in work:
        for index, record in enumerate(pending, start=1):
            try:
                results[record["alias"]] = generate_phrases(
                    args.model, args.ollama_url, record, args.timeout, args.retries
                )
                errors.pop(record["alias"], None)
            except ValueError as error:
                errors[record["alias"]] = str(error)
                save_results(output_path, results)
                save_results(errors_path, errors)
                print(f"[ERROR] {record['alias']}: {error}; continúo con el siguiente registro")
            except (OSError, RuntimeError) as error:
                errors[record["alias"]] = str(error)
                save_results(output_path, results)
                save_results(errors_path, errors)
                raise RuntimeError(
                    f"Falló {record['alias']}; resultados previos guardados en "
                    f"{output_path}. Reejecuta con --resume para reintentar este alias: {error}"
                ) from error

            save_results(output_path, results)
            save_results(errors_path, errors)
            if index % 10 == 0 or index == len(pending):
                print(
                    f"{corpus_file.name}: procesados {index}/{len(pending)}; "
                    f"último alias: {record['alias']}"
                )

        if not output_path.exists():
            save_results(output_path, results)
        save_results(errors_path, errors)
        print(
            f"Resultado guardado en {output_path} ({len(results)} aliases); "
            f"fallidos: {len(errors)}"
        )


if __name__ == "__main__":
    main()
