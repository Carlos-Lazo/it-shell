from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class QueryRequest:
    query: str
    retrieval_top_k: int = 50
    output_format: str = "cli"

@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    normalized: str
    language: str
    is_complex: bool

@dataclass(frozen=True)
class PlanStep:
    id: str
    query: str
    depends_on: tuple[str, ...] = ()
    arguments: tuple[tuple[str, str], ...] = ()
    connector_before: str | None = None
    redirect_operator: str | None = None
    redirect_target: str | None = None
    ambiguous_arguments: tuple[str, ...] = ()

@dataclass(frozen=True)
class QueryPlan:
    goal: str
    steps: tuple[PlanStep, ...]

@dataclass(frozen=True)
class CandidateCommand:
    alias: str
    category: str
    source_id: int
    intent: str
    command: str
    example: str
    retrieval_score: float
    reranker_score: float | None = None

    @property
    def score(self) -> float:
        return self.reranker_score if self.reranker_score is not None else self.retrieval_score

@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    step_id: str | None = None
    recommendation: str = ""

@dataclass(frozen=True)
class PipelineStep:
    id: str
    query: str
    depends_on: tuple[str, ...]
    selected: CandidateCommand | None
    alternatives: tuple[CandidateCommand, ...] = ()
    command_template: str | None = None
    ambiguous_arguments: tuple[str, ...] = ()
    connector_before: str | None = None
    redirect_operator: str | None = None
    redirect_target: str | None = None

@dataclass(frozen=True)
class PipelineProposal:
    goal: str
    steps: tuple[PipelineStep, ...]
    requires_input: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    executable: bool = False

@dataclass(frozen=True)
class SearchResponse:
    analysis: QueryAnalysis
    plan: QueryPlan
    proposal: PipelineProposal
    clarification_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
