import re
from semantic_search_engine.shared.contracts import PlanStep, QueryAnalysis, QueryPlan

SEQUENCE_SPLITTER = re.compile(
    r'\s*(?:&&|\|{1,2}|;)\s*'
    r'|'
    r'\s+(?:'
    r'(?:and\s+)?(?:then|next|after\s+that|afterwards|followed\s+by)'
    r'|'
    r'(?:y\s+)?(?:luego|después|despues|a\s+continuación|a\s+continuacion|seguido\s+de)'
    r')\s+',
    re.IGNORECASE,
)

class RulePlanner:
    """Divide secuencias explícitas sin inventar comandos ni argumentos."""
    splitter = SEQUENCE_SPLITTER

    def plan(self, analysis: QueryAnalysis) -> QueryPlan:
        queries = [part.strip(' ,.;') for part in self.splitter.split(analysis.normalized) if part.strip(' ,.;')]
        queries = queries or [analysis.normalized]
        steps = tuple(
            PlanStep(f'step_{i}', query, (f'step_{i-1}',) if i > 1 else ())
            for i, query in enumerate(queries, 1)
        )
        return QueryPlan(analysis.normalized, steps)
