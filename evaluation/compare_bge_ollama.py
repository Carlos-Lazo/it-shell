#!/usr/bin/env python3
"""Compare BGE-M3 retrieval with deterministic rules versus local Ollama."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import faiss
import numpy as np

from evaluation.evaluate_capacity_retrieval import (
    load_capacity_queries,
    load_source_commands,
    sha256_file,
)
from semantic_search_engine.llm.ollama_assistant import (
    CandidateOption,
    OllamaAssistant,
)
from semantic_search_engine.pipeline_builder_service.command_resolver import CommandResolver
from semantic_search_engine.retrieval_service.retriever import (
    FaissRetriever,
    SentenceTransformerEncoder,
)

ROOT = Path(__file__).resolve().parents[1]
CAPACITIES = ROOT / "evaluation/capacity/plain_source.yaml"
CORPUS = ROOT / "dataset_builder/corpus"
BGE_INDEX = ROOT / "dataset_builder/indexer/artifacts/bge-m3/index.faiss"
BGE_ALIASES = ROOT / "dataset_builder/indexer/artifacts/bge-m3/aliases.npy"
OUTPUT_JSON = ROOT / "evaluation/capacity/bge_ollama_comparison.json"
OUTPUT_MARKDOWN = ROOT / "evaluation/capacity/bge_ollama_comparison.md"
BGE_MODEL = "BAAI/bge-m3"
OLLAMA_MODEL = "qwen2.5:1.5b"
RETRIEVAL_TOP_K = 15
METRIC_KS = (1, 5, 10, 15)
MIN_RETRIEVAL_SCORE = 0.45
SELECTION_CANDIDATE_TOP_K = 5
logger = logging.getLogger(__name__)
BENCHMARK_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_index": {"type": ["integer", "null"]},
    },
    "required": ["selected_index"],
    "additionalProperties": False,
}


class RecordingOllamaAssistant(OllamaAssistant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_calls = 0
        self.api_time_s = 0.0
        self.current_trace: dict[str, Any] = {}

    def _chat_request(self, *args, **kwargs):
        started = time.perf_counter()
        self.api_calls += 1
        try:
            return super()._chat_request(*args, **kwargs)
        finally:
            self.api_time_s += time.perf_counter() - started

    def analyze_query(self, query, allowed_categories=("filesystem", "networking")):
        raise RuntimeError("This controlled benchmark does not call query analysis")

    def select_candidate(
        self,
        query: str,
        candidates: list[CandidateOption],
        language: str,
    ) -> str | None:
        if not candidates:
            return None
        payload = {
            "user_request": query,
            "candidates": [
                {
                    "index": candidate.index,
                    "alias": candidate.alias,
                    "command": candidate.command[:120],
                    "intent": candidate.intent[:180],
                }
                for candidate in candidates
            ],
        }
        system_content = (
            "Elegí el candidato que mejor resuelve la solicitud del usuario. "
            "Elegí únicamente entre los candidatos dados. Devolvé el índice "
            "basado en 1, o null si ninguno corresponde."
            if language == "es"
            else
            "Choose the single candidate that best fulfills the user request. "
            "Select only from the provided candidates. Return its 1-based "
            "index, or null if none fits."
        )
        try:
            response = self._chat_request(
                [
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                schema=BENCHMARK_SELECTION_SCHEMA,
                temperature=0.0,
                num_predict=24,
            )
        except Exception as error:
            self.selection_fallback_count += 1
            logger.warning(
                "Ollama candidate selection failed (%s); using the top eligible candidate.",
                error,
            )
            return candidates[0].alias
        selected_index = response.get("selected_index")
        if selected_index is not None and type(selected_index) is not int:
            self.selection_fallback_count += 1
            logger.warning(
                "Ollama returned a non-integer candidate index; "
                "using the top eligible candidate."
            )
            return candidates[0].alias
        selected = next(
            (candidate for candidate in candidates if candidate.index == selected_index),
            None,
        )
        if selected is None and selected_index is not None:
            self.selection_fallback_count += 1
            return candidates[0].alias
        self.current_trace["raw_response"] = response
        return selected.alias if selected else None

def ollama_metadata(base_url: str, model_name: str) -> dict[str, Any]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/version", timeout=3) as response:
            version_payload = json.load(response)
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as response:
            tags_payload = json.load(response)
    except Exception as error:
        raise RuntimeError(f"Could not inspect local Ollama at {base_url}: {error}") from error

    model = next(
        (entry for entry in tags_payload.get("models", []) if entry.get("name") == model_name),
        None,
    )
    if model is None:
        available = ", ".join(
            entry.get("name", "<unnamed>")
            for entry in tags_payload.get("models", [])
        )
        raise RuntimeError(
            f"Model {model_name!r} is not available in Ollama. Available: {available}"
        )
    if model.get("name") != model_name:
        raise RuntimeError(
            f"Requested model {model_name!r}, but Ollama reports {model.get('name')!r}"
        )
    return {
        "version": version_payload.get("version"),
        "model": model.get("name"),
        "digest": model.get("digest"),
        "size_bytes": model.get("size"),
    }


def validate_bge_index(index_path: Path, aliases_path: Path):
    index = faiss.read_index(str(index_path))
    aliases = np.load(aliases_path, allow_pickle=False).astype(str)
    if index.ntotal != len(aliases) or len(set(aliases)) != len(aliases):
        raise ValueError("BGE-M3 index and unique aliases are not aligned")
    if index.d != 1024 or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Expected normalized 1024-d BGE-M3 inner-product index")
    return index, aliases


def query_metrics(
    ranked_aliases: list[str],
    selected_command: str | None,
    relevant_commands: set[str],
    source_commands: dict[str, str],
) -> dict[str, float]:
    hits = [source_commands.get(alias) in relevant_commands for alias in ranked_aliases]
    relevant_rank = next(
        (rank for rank, hit in enumerate(hits, start=1) if hit),
        None,
    )
    metrics: dict[str, float] = {
        "selected_command_correct": float(selected_command in relevant_commands),
        "mrr": 1 / relevant_rank if relevant_rank else 0.0,
    }
    for k in METRIC_KS:
        metrics[f"command_hit@{k}"] = float(any(hits[:k]))
    metrics["selected_given_hit@15"] = float(
        selected_command in relevant_commands and any(hits[:15])
    )
    metrics["candidate_hit@15"] = float(any(hits[:15]))
    return metrics


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"queries": 0}
    summary: dict[str, Any] = {"queries": len(records)}
    for metric in (
        "selected_command_correct",
        "selected_given_hit@15",
        "candidate_hit@15",
        "mrr",
        *(f"command_hit@{k}" for k in METRIC_KS),
    ):
        summary[metric] = round(
            sum(record[metric] for record in records) / len(records),
            4,
        )
    summary["mean_latency_seconds"] = round(
        sum(record["latency_seconds"] for record in records) / len(records),
        3,
    )
    return summary


def grouped_summaries(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record[field]].append(record)
    return {
        key: summarize_records(group)
        for key, group in sorted(groups.items())
    }


def evaluate_shared_retrieval(
    queries: list[dict[str, Any]],
    source_commands: dict[str, str],
    retriever: FaissRetriever,
    resolver: CommandResolver,
    assistant: RecordingOllamaAssistant,
    checkpoint_path: Path,
    run_fingerprint: str,
) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {"rules": [], "ollama": []}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"Existing checkpoint has different inputs: {checkpoint_path}; "
                "move it aside before starting a new benchmark"
            )
        records = checkpoint["records"]
    processed_ids = {record["id"] for record in records["rules"]}

    for query in queries:
        if query["id"] in processed_ids:
            continue
        retrieve_started = time.perf_counter()
        retrieved = retriever.retrieve(query["query"], RETRIEVAL_TOP_K)
        retrieval_seconds = time.perf_counter() - retrieve_started
        candidates = [
            resolver.resolve(alias, score)
            for alias, score in retrieved
        ]
        eligible = [
            candidate
            for candidate in candidates
            if candidate.retrieval_score >= MIN_RETRIEVAL_SCORE
        ][:SELECTION_CANDIDATE_TOP_K]
        relevant = set(query["relevant_commands"])
        ranked_aliases = [alias for alias, _ in retrieved]
        rules_selected = eligible[0] if eligible else None
        for method, selected in (
            ("rules", rules_selected),
            ("ollama", None),
        ):
            if method == "ollama":
                assistant.current_trace = {}
                options = [
                    CandidateOption(
                        index=position,
                        alias=candidate.alias,
                        command=candidate.example or candidate.command,
                        intent=candidate.intent,
                        example=candidate.example,
                        score=candidate.score,
                    )
                    for position, candidate in enumerate(eligible, start=1)
                ]
                selection_started = time.perf_counter()
                calls_before = assistant.api_calls
                api_time_before = assistant.api_time_s
                fallbacks_before = assistant.selection_fallback_count
                selected_alias = assistant.select_candidate(
                    query["query"], options, query["language"]
                )
                selection_seconds = time.perf_counter() - selection_started
                selected = next(
                    (
                        candidate
                        for candidate in eligible
                        if candidate.alias == selected_alias
                    ),
                    None,
                )
                assistant.current_trace["selection"] = {
                    "returned_alias": selected_alias,
                    "accepted_alias": selected.alias if selected else None,
                    "confidence": "structured-index-only",
                    "raw_response": assistant.current_trace.pop("raw_response", None),
                }
                assistant.current_trace["chat_calls"] = (
                    assistant.api_calls - calls_before
                )
                assistant.current_trace["api_time_seconds"] = round(
                    assistant.api_time_s - api_time_before, 3
                )
                assistant.current_trace["fallback_count"] = (
                    assistant.selection_fallback_count - fallbacks_before
                )
            else:
                selection_seconds = 0.0

            selected_command = (
                source_commands.get(selected.alias) if selected else None
            )
            metrics = query_metrics(
                ranked_aliases,
                selected_command,
                relevant,
                source_commands,
            )
            records[method].append({
                "id": query["id"],
                "query": query["query"],
                "language": query["language"],
                "partition": query["partition"],
                "category": query["category"],
                "relevant_commands": sorted(relevant),
                "retrieval_top_aliases": ranked_aliases[:10],
                "retrieval_first_relevant_rank": next(
                    (
                        rank
                        for rank, alias in enumerate(ranked_aliases, start=1)
                        if source_commands.get(alias) in relevant
                    ),
                    None,
                ),
                "selected_alias": selected.alias if selected else None,
                "selected_command": selected_command,
                "latency_seconds": round(
                    retrieval_seconds + selection_seconds, 3
                ),
                "retrieval_latency_seconds": round(retrieval_seconds, 3),
                "selection_latency_seconds": round(selection_seconds, 3),
                "assistant_trace": (
                    dict(assistant.current_trace) if method == "ollama" else None
                ),
                **metrics,
            })
        print(
            f"Processed {len(processed_ids) + 1}/{len(queries)}: {query['id']}",
            flush=True,
        )
        processed_ids.add(query["id"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        checkpoint_temp.write_text(
            json.dumps({
                "run_fingerprint": run_fingerprint,
                "records": records,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        checkpoint_temp.replace(checkpoint_path)

    return {
        method: {
            "summary": summarize_records(method_records),
            "by_partition": grouped_summaries(method_records, "partition"),
            "by_language": grouped_summaries(method_records, "language"),
            "by_partition_and_language": {
                f"{partition}/{language}": summarize_records([
                    record
                    for record in method_records
                    if record["partition"] == partition
                    and record["language"] == language
                ])
                for partition in ("development", "test")
                for language in ("en", "es")
                if any(
                    record["partition"] == partition
                    and record["language"] == language
                    for record in method_records
                )
            },
            "queries": method_records,
        }
        for method, method_records in records.items()
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Comparación BGE-M3: reglas vs. Ollama",
        "",
        f"- Consultas: {report['query_counts']['total']} "
        f"({report['query_counts']['development']} development, "
        f"{report['query_counts']['test']} test).",
        f"- Índice: {report['provenance']['index_vectors']} vectores BGE-M3, "
        f"{report['provenance']['embedding_dimension']} dimensiones.",
        f"- Ollama: `{report['provenance']['ollama']['version']}`, "
        f"`{report['provenance']['ollama']['model']}`, digest "
        f"`{report['provenance']['ollama']['digest']}`.",
        f"- Ollama candidate-selection calls: "
        f"{report['methods']['ollama']['assistant_chat_calls']}; "
        f"selection fallbacks: {report['methods']['ollama']['fallbacks']['selection']}.",
        "- El test se reporta con parámetros fijos; no se usó para seleccionar configuración.",
        "- La métrica de selección evalúa el comando del alias elegido; "
        "no valida todos sus argumentos ni ejecuta comandos.",
        "",
        "## Resultado general",
        "",
        "| Método | BGE Hit@1 | BGE Hit@5 | BGE Hit@10 | BGE MRR | Selección correcta | Latencia media |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("rules", "ollama"):
        metrics = report["methods"][method]["summary"]
        lines.append(
            f"| {method} | {metrics['command_hit@1']:.3f} | "
            f"{metrics['command_hit@5']:.3f} | {metrics['command_hit@10']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['selected_command_correct']:.3f} | "
            f"{metrics['mean_latency_seconds']:.2f} s |"
        )

    lines.extend([
        "",
        "## Por partición e idioma",
        "",
        "| Partición/idioma | Método | Consultas | Hit@10 | MRR | Selección final correcta |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for group in ("development/en", "development/es", "test/en", "test/es"):
        for method in ("rules", "ollama"):
            metrics = report["methods"][method]["by_partition_and_language"].get(group)
            if metrics:
                lines.append(
                    f"| {group} | {method} | {metrics['queries']} | "
                    f"{metrics['command_hit@10']:.3f} | {metrics['mrr']:.3f} | "
                    f"{metrics['selected_command_correct']:.3f} |"
                )
    lines.extend([
        "",
        "## Procedencia y límites",
        "",
        "Esta primera comparación aísla la selección final: ambos métodos reciben "
        "el mismo ranking BGE-M3 para la consulta original. Rules selecciona el "
        "primer candidato que supera el umbral; Ollama selecciona entre ese mismo "
        f"conjunto de hasta {SELECTION_CANDIDATE_TOP_K} candidatos elegibles. "
        "No se evalúan reformulación de consultas, análisis "
        "de pasos ni calidad de explicación. El comando ensamblado por Ollama se "
        "ignora; la métrica cuenta únicamente si el comando documentado del alias "
        "seleccionado coincide con una etiqueta relevante.",
        "",
        report["provenance"]["bge_corpus_note"],
        "",
        "El retrieval es compartido, por lo que Hit@k y MRR describen BGE-M3 y son "
        "idénticos para ambas condiciones. La comparación del selector usa "
        "selected_command_correct. El benchmark mide acierto del comando seleccionado, "
        "no ejecución funcional ni corrección integral de argumentos. Las métricas por "
        "idioma describen esta muestra pequeña de capacidades.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-timeout", type=float, default=45.0)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.top_k != RETRIEVAL_TOP_K:
        parser.error(f"--top-k is fixed at {RETRIEVAL_TOP_K} for this comparison")
    if args.ollama_timeout <= 0:
        parser.error("--ollama-timeout must be positive")

    ollama_info = ollama_metadata(args.ollama_url, args.model)
    index, aliases = validate_bge_index(BGE_INDEX, BGE_ALIASES)
    indexed_aliases = set(aliases)
    queries = load_capacity_queries(CAPACITIES, CORPUS, indexed_aliases)
    if len(queries) != 122:
        raise ValueError(f"Expected 122 reviewed capacity queries, got {len(queries)}")
    source_commands = load_source_commands(CORPUS)
    for query in queries:
        missing = [
            command for command in query["relevant_commands"]
            if not any(
                alias in indexed_aliases and command_name == command
                for alias, command_name in source_commands.items()
            )
        ]
        if missing:
            raise ValueError(f"{query['id']}: relevant commands absent from BGE: {missing}")

    corpus_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(CORPUS.glob("*.yaml"))
    }
    run_fingerprint = hashlib.sha256(json.dumps({
        "ollama": ollama_info,
        "embedding_model": BGE_MODEL,
        "index_sha256": sha256_file(BGE_INDEX),
        "aliases_sha256": sha256_file(BGE_ALIASES),
        "capacities_sha256": sha256_file(CAPACITIES),
        "corpus_sha256": corpus_hashes,
        "top_k": args.top_k,
        "min_retrieval_score": MIN_RETRIEVAL_SCORE,
        "selection_candidate_top_k": SELECTION_CANDIDATE_TOP_K,
        "selection_schema": BENCHMARK_SELECTION_SCHEMA,
    }, sort_keys=True).encode("utf-8")).hexdigest()

    encoder = SentenceTransformerEncoder(BGE_MODEL)
    retriever = FaissRetriever(BGE_INDEX, BGE_ALIASES, encoder)
    resolver = CommandResolver(CORPUS)
    ollama = RecordingOllamaAssistant(
        model=args.model,
        base_url=args.ollama_url,
        timeout=args.ollama_timeout,
        retries=0,
    )

    development = [query for query in queries if query["partition"] == "development"]
    test = [query for query in queries if query["partition"] == "test"]
    if len(development) != 86 or len(test) != 36:
        raise ValueError("Expected fixed capacity split of 86 development and 36 test")

    results = evaluate_shared_retrieval(
        queries,
        source_commands,
        retriever,
        resolver,
        ollama,
        args.checkpoint or args.output_json.with_suffix(".selector.partial.json"),
        run_fingerprint,
    )
    chat_calls = sum(
        record["assistant_trace"]["chat_calls"]
        for record in results["ollama"]["queries"]
    )
    selection_fallbacks = sum(
        record["assistant_trace"]["fallback_count"]
        for record in results["ollama"]["queries"]
    )
    assistant_api_time_seconds = round(sum(
        record["assistant_trace"]["api_time_seconds"]
        for record in results["ollama"]["queries"]
    ), 3)

    report = {
        "schema_version": 1,
        "kind": "bge_m3_rules_vs_ollama",
        "query_counts": {
            "development": len(development),
            "test": len(test),
            "total": len(queries),
        },
        "language_query_counts": {
            language: sum(query["language"] == language for query in queries)
            for language in ("en", "es")
        },
        "configuration": {
            "embedding_model": BGE_MODEL,
            "retrieval_top_k": args.top_k,
            "selection_candidate_top_k": SELECTION_CANDIDATE_TOP_K,
            "selection_min_score": MIN_RETRIEVAL_SCORE,
            "rules_selector": "first candidate above retrieval threshold",
            "ollama_selector": "single structured selection call over same candidates",
            "ollama_retries": 0,
            "query_rewriting_enabled": False,
            "explanation_generation_enabled": False,
            "retrieval_pool_shared_between_methods": True,
            "test_used_for_selection": False,
            "assistant_fallback_policy": "record fallback and preserve top eligible candidate",
        },
        "methods": {
            **results,
            "ollama": {
                **results["ollama"],
                "assistant_chat_calls": chat_calls,
                "fallbacks": {
                    "selection": selection_fallbacks,
                },
                "assistant_api_time_seconds": assistant_api_time_seconds,
            },
        },
        "provenance": {
            "ollama": ollama_info,
            "embedding_model": BGE_MODEL,
            "embedding_dimension": index.d,
            "index_vectors": index.ntotal,
            "index_sha256": sha256_file(BGE_INDEX),
            "aliases_sha256": sha256_file(BGE_ALIASES),
            "capacities_sha256": sha256_file(CAPACITIES),
            "corpus_sha256": corpus_hashes,
            "bge_vectors_sha256": sha256_file(
                ROOT / "dataset_builder/embedder/artifacts/bge_m3_vectors.npz"
            ),
            "bge_corpus_note": (
                "The BGE-M3 aliases align to the current indexed corpus subset, "
                "but the BGE vector artifact does not include hashes of source text. "
                "Exact source-text revision cannot be independently verified."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    (args.checkpoint or args.output_json.with_suffix(".selector.partial.json")).unlink(
        missing_ok=True
    )
    print(json.dumps({
        "query_counts": report["query_counts"],
        "language_query_counts": report["language_query_counts"],
        "ollama": ollama_info,
        "rules": results["rules"]["summary"],
        "ollama_metrics": results["ollama"]["summary"],
        "ollama_fallbacks": selection_fallbacks,
        "ollama_chat_calls": chat_calls,
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
