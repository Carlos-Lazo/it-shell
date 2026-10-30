from dataclasses import asdict
from typing import Any

from semantic_search_engine.shared.contracts import (
    CandidateCommand,
    PipelineProposal,
    PipelineStep,
    QueryRequest,
)

MAX_QUERY_LENGTH = 2_000
MAX_COMMAND_LENGTH = 4_096
MAX_TOP_K = 50
DEFAULT_TOP_K = 10


def _validate_text(value: str, name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} debe ser texto")
    if len(value) > max_length:
        raise ValueError(f"{name} supera el máximo de {max_length} caracteres")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} no puede estar vacío")
    return text


class MCPToolService:
    """Implements the MCP tools using the existing engine services."""

    def __init__(self, engine):
        self.engine = engine

    def search_commands(self, query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        query = _validate_text(query, "query", MAX_QUERY_LENGTH)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k debe ser un entero entre 1 y {MAX_TOP_K}")

        _, plan, retrieved = self.engine.retrieval.retrieve_plan(query, top_k)
        steps = []
        for step in plan.steps:
            candidates = [
                self.engine.pipeline.resolver.resolve(alias, score)
                for alias, score in retrieved[step.id]
            ]
            ranked = self.engine.reranker.rerank(step.query, candidates)
            steps.append({
                "id": step.id,
                "query": step.query,
                "candidates": [
                    {**asdict(candidate), "score": candidate.score}
                    for candidate in ranked
                ],
            })
        return {"query": query, "steps": steps}

    def plan_operation(self, query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        query = _validate_text(query, "query", MAX_QUERY_LENGTH)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k debe ser un entero entre 1 y {MAX_TOP_K}")
        return self.engine.search_response(
            QueryRequest(query=query, retrieval_top_k=top_k)
        ).to_dict()

    def validate_command(self, command: str) -> dict[str, Any]:
        command = _validate_text(command, "command", MAX_COMMAND_LENGTH)
        candidate = CandidateCommand(
            alias="user:provided",
            category="user",
            source_id=0,
            intent="Comando proporcionado para análisis de riesgo",
            command=command,
            example=command,
            retrieval_score=1.0,
        )
        step = PipelineStep(
            id="step_1",
            query=command,
            depends_on=(),
            selected=candidate,
            command_template=command,
        )
        proposal = self.engine.pipeline.validator.validate(
            PipelineProposal(goal=command, steps=(step,))
        )
        return asdict(proposal)
