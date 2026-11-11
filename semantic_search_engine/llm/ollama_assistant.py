"""Ollama assistant for query analysis, category classification, candidate selection and explanation.

Designed to be lightweight and robust with small models such as qwen2.5:1.5b.
Operates via standard HTTP JSON endpoints, with strict schema validation and
deterministic fallback mechanisms.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from semantic_search_engine.prompt_builder_service.llm_client import LLMClientError

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_CATEGORIES = ("filesystem", "networking")


class OllamaConnectionError(Exception):
    """Raised when Ollama server cannot be reached or times out."""


class OllamaResponseError(Exception):
    """Raised when Ollama returns an invalid or unparseable response."""


@dataclass
class AnalysisStep:
    """A single sub-task or step extracted from the user query."""

    step_index: int
    task: str
    search_query: str
    arguments: dict[str, Any] = field(default_factory=dict)
    clarification_needed: bool = False
    clarification_question: str | None = None


@dataclass
class LLMQueryAnalysis:
    """Structured query analysis produced by the LLM."""

    original_query: str
    predicted_category: str | None
    category_confidence: str  # "high", "medium", "low", "none"
    language: str  # "es", "en", etc.
    steps: list[AnalysisStep] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None

    @property
    def is_confident_category(self) -> bool:
        """True if category prediction is high-confidence and non-null."""
        return bool(self.predicted_category and self.category_confidence == "high")


@dataclass
class CandidateOption:
    """A retrieved command candidate presented to the LLM for selection."""

    index: int
    alias: str
    command: str
    intent: str
    example: str = ""
    score: float = 0.0


@dataclass
class LLMSelectionResult:
    """Candidate selection and human synthesis produced by the LLM."""

    selected_index: int | None
    selected_alias: str | None
    confidence: str  # "high", "medium", "low", "none"
    assembled_command: str | None
    explanation: str
    issues_or_warnings: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None


QUERY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_category": {
            "type": ["string", "null"],
            "description": "Category name if clearly matching, else null.",
        },
        "category_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
        },
        "language": {
            "type": "string",
            "enum": ["es", "en", "other"],
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer"},
                    "task": {"type": "string"},
                    "search_query": {"type": "string"},
                    "arguments": {"type": "object"},
                    "clarification_needed": {"type": "boolean"},
                    "clarification_question": {"type": ["string", "null"]},
                },
                "required": ["step_index", "task", "search_query"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
    },
    "required": ["predicted_category", "category_confidence", "language", "steps"],
    "additionalProperties": False,
}

SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_index": {
            "type": ["integer", "null"],
            "description": "1-based index of candidate chosen, or null if none matches.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "none"],
        },
        "assembled_command": {
            "type": ["string", "null"],
            "description": "Selected command with user arguments applied.",
        },
        "explanation": {
            "type": "string",
            "description": "Concise 1-2 sentence explanation in the user's language.",
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["selected_index", "confidence", "assembled_command", "explanation"],
    "additionalProperties": False,
}

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
    "additionalProperties": False,
}


class OllamaAssistant:
    """Interacts with an Ollama model for query analysis and candidate reranking/synthesis."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 45.0,
        retries: int = 1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.analysis_fallback_count = 0
        self.selection_fallback_count = 0

    def generate(self, prompt: str) -> str:
        """Generate a grounded explanation for the validated proposal."""
        try:
            parsed = self._chat_request(
                [
                    {
                        "role": "system",
                        "content": (
                            "Explica brevemente la propuesta descrita en el contexto. "
                            "No modifiques ni completes comandos. Devuelve solo el "
                            "campo explanation del esquema JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                schema=EXPLANATION_SCHEMA,
                temperature=0.1,
            )
        except (OllamaConnectionError, OllamaResponseError) as error:
            raise LLMClientError(f"Ollama explanation failed: {error}") from error
        explanation = parsed.get("explanation")
        if not isinstance(explanation, str):
            raise LLMClientError("Ollama returned no explanation string.")
        return explanation

    def is_available(self) -> bool:
        """Check if the Ollama endpoint is reachable."""
        try:
            req = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _chat_request(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
        num_predict: int = 160,
    ) -> dict[str, Any]:
        """Perform a chat completion with JSON schema formatting."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 2048,
                "num_predict": num_predict,
            },
        }
        if schema:
            payload["format"] = schema
        else:
            payload["format"] = "json"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    raw = json.load(resp)
                    if not isinstance(raw, dict):
                        raise OllamaResponseError("Ollama response was not a JSON dictionary.")
                    message = raw.get("message", {})
                    content_text = message.get("content", "")
                    try:
                        return json.loads(content_text)
                    except json.JSONDecodeError as exc:
                        raise OllamaResponseError(
                            f"Model content is not valid JSON: {content_text[:200]}"
                        ) from exc
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise OllamaResponseError(f"HTTP {exc.code} from Ollama: {detail}") from exc
            except (TimeoutError, URLError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.0)
                    continue
                raise OllamaConnectionError(
                    f"Could not connect to Ollama at {self.base_url} ({exc})"
                ) from exc

        raise OllamaConnectionError(f"Ollama request failed after retries: {last_error}")

    def analyze_query(
        self,
        query: str,
        allowed_categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    ) -> LLMQueryAnalysis:
        """Decompose a user query into structured steps, arguments, and predicted category.

        Parameters
        ----------
        query:
            The raw user query (e.g. in Spanish or English).
        allowed_categories:
            Known corpus categories to classify into (e.g. filesystem, networking).
        """
        system_prompt = (
            "You are an expert Linux shell assistant specialized in query understanding.\n"
            "Analyze the user's command-line request.\n"
            f"1. Classify the request into one of these categories: {list(allowed_categories)}.\n"
            "   If the query does not clearly fit any of them, set predicted_category to null and confidence to 'low'.\n"
            "2. Identify the language ('es' for Spanish, 'en' for English).\n"
            "3. Break the query into 1 or more sequential steps.\n"
            "   For each step: extract a concise 'search_query' (in English keywords suitable for man page retrieval),\n"
            "   a 'task' description, and any concrete 'arguments' (paths, files, ports, flags) mentioned.\n"
            "   If critical arguments are completely missing and the request is ambiguous, set clarification_needed to true.\n"
            "Return strictly the requested JSON schema without any commentary."
        )

        user_content = json.dumps({"query": query}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            parsed = self._chat_request(messages, schema=QUERY_ANALYSIS_SCHEMA, temperature=0.1)
        except Exception as exc:
            self.analysis_fallback_count += 1
            logger.warning("Ollama query analysis failed (%s); using deterministic fallback.", exc)
            return self.fallback_analysis(query)

        steps: list[AnalysisStep] = []
        raw_steps = parsed.get("steps") or []
        for i, s in enumerate(raw_steps, start=1):
            steps.append(
                AnalysisStep(
                    step_index=s.get("step_index", i),
                    task=str(s.get("task", query)),
                    search_query=str(s.get("search_query", query)),
                    arguments=s.get("arguments") if isinstance(s.get("arguments"), dict) else {},
                    clarification_needed=bool(s.get("clarification_needed", False)),
                    clarification_question=s.get("clarification_question"),
                )
            )

        cat = parsed.get("predicted_category")
        if cat and cat not in allowed_categories:
            cat = None

        q_lower = query.lower()
        is_spanish = any(
            w in q_lower
            for w in (
                "crear", "buscar", "como", "listar", "ver", "mostrar", "red",
                "carpeta", "carpetas", "archivo", "archivos", "borrar", "eliminar",
                "puerto", "puertos", "anidadas", "padre", "padres", "si", "no", "para"
            )
        )
        detected_lang = "es" if is_spanish else parsed.get("language", "en")

        return LLMQueryAnalysis(
            original_query=query,
            predicted_category=cat,
            category_confidence=parsed.get("category_confidence", "none"),
            language=detected_lang,
            steps=steps or [AnalysisStep(step_index=1, task=query, search_query=query)],
            raw_response=parsed,
        )

    def select_and_synthesize(
        self,
        query: str,
        step_task: str,
        candidates: list[CandidateOption],
        arguments: dict[str, Any] | None = None,
        language: str = "es",
    ) -> LLMSelectionResult:
        """Select the best candidate among retrieved options and generate a human explanation.

        Parameters
        ----------
        query:
            Original user query.
        step_task:
            The specific sub-task being resolved.
        candidates:
            List of CandidateOption retrieved from corpus/index.
        arguments:
            Extracted arguments to populate in the command.
        language:
            Target language for the explanation ('es' or 'en').
        """
        if not candidates:
            return LLMSelectionResult(
                selected_index=None,
                selected_alias=None,
                confidence="none",
                assembled_command=None,
                explanation="No se encontraron comandos candidatos para esta tarea."
                if language == "es"
                else "No candidate commands found for this task.",
                issues_or_warnings=["No candidates provided."],
            )

        candidates_data = [
            {
                "index": c.index,
                "alias": c.alias,
                "command": c.command,
                "intent": c.intent,
                "example": c.example,
            }
            for c in candidates
        ]

        if language == "es":
            system_prompt = (
                "Eres un administrador de sistemas experto en Linux y selección de comandos.\n"
                "Dada una consulta de usuario y una lista de comandos candidatos de páginas man de Debian,\n"
                "selecciona el ÍNDICE (1-based) del candidato que mejor resuelve la tarea.\n"
                "- Si ninguno resuelve la tarea con seguridad, indica selected_index: null.\n"
                "- Si un candidato coincide, ensambla el comando incorporando los argumentos del usuario.\n"
                "- Escribe una explicación breve de 1 o 2 oraciones OBLIGATORIAMENTE EN ESPAÑOL.\n"
                "- Agrega advertencias si el comando borra o modifica archivos del sistema.\n"
                "Devuelve estrictamente el esquema JSON solicitado."
            )
        else:
            system_prompt = (
                "You are an expert Linux sysadmin and command selector.\n"
                "Given a user query and a list of documented candidate commands from Debian man pages,\n"
                "select the SINGLE candidate index (1-based) that best satisfies the task.\n"
                "- If none of the candidates fulfill the task safely, set selected_index to null.\n"
                "- If a candidate matches, assemble the command incorporating the user's arguments if provided.\n"
                "- Write a concise, natural 1-2 sentence explanation strictly in English.\n"
                "- Add warnings if the command modifies or removes files.\n"
                "Return strictly the requested JSON schema."
            )

        user_payload = {
            "user_query": query,
            "task": step_task,
            "arguments": arguments or {},
            "candidates": candidates_data,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        try:
            parsed = self._chat_request(messages, schema=SELECTION_SCHEMA, temperature=0.1)
        except Exception as exc:
            self.selection_fallback_count += 1
            logger.warning("Ollama candidate selection failed (%s); using top candidate fallback.", exc)
            top = candidates[0]
            return LLMSelectionResult(
                selected_index=top.index,
                selected_alias=top.alias,
                confidence="low",
                assembled_command=top.command,
                explanation=f"Seleccionado por similitud léxica/semántica: {top.intent}",
                issues_or_warnings=[f"Fallback to top candidate due to LLM error: {exc}"],
            )

        chosen_idx = parsed.get("selected_index")
        chosen_alias = None
        if chosen_idx is not None and isinstance(chosen_idx, int):
            for c in candidates:
                if c.index == chosen_idx:
                    chosen_alias = c.alias
                    break

        return LLMSelectionResult(
            selected_index=chosen_idx,
            selected_alias=chosen_alias,
            confidence=parsed.get("confidence", "medium"),
            assembled_command=parsed.get("assembled_command"),
            explanation=parsed.get("explanation", ""),
            issues_or_warnings=parsed.get("warnings", []),
            raw_response=parsed,
        )

    def fallback_analysis(self, query: str) -> LLMQueryAnalysis:
        """Deterministic rule-based fallback when Ollama is unavailable."""
        q_lower = query.lower()
        is_spanish = any(w in q_lower for w in ("crear", "buscar", "como", "listar", "ver", "mostrar", "red"))

        # Simple category heuristic for fallback
        fs_keywords = ("archivo", "carpeta", "directorio", "file", "folder", "directory", "mkdir", "rm", "cp", "tar", "grep", "find")
        net_keywords = ("red", "puerto", "ip", "interface", "ping", "socket", "network", "port", "ss", "curl", "netstat", "route")

        if any(k in q_lower for k in fs_keywords):
            cat = "filesystem"
            conf = "medium"
        elif any(k in q_lower for k in net_keywords):
            cat = "networking"
            conf = "medium"
        else:
            cat = None
            conf = "none"

        return LLMQueryAnalysis(
            original_query=query,
            predicted_category=cat,
            category_confidence=conf,
            language="es" if is_spanish else "en",
            steps=[AnalysisStep(step_index=1, task=query, search_query=query)],
            raw_response={"fallback": True},
        )
