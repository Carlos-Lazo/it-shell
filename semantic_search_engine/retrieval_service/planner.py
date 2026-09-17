import re
from semantic_search_engine.shared.contracts import PlanStep, QueryAnalysis, QueryPlan

class RulePlanner:
    """Divide secuencias explícitas sin inventar comandos ni argumentos."""
    splitter=re.compile(r'\s+(?:y luego|después|luego|and then|then)\s+', re.I)
    def plan(self, analysis: QueryAnalysis) -> QueryPlan:
        queries=[part.strip(' ,.;') for part in self.splitter.split(analysis.normalized) if part.strip(' ,.;')]
        queries=queries or [analysis.normalized]
        steps=tuple(PlanStep(f'step_{i}', query, (f'step_{i-1}',) if i>1 else ()) for i, query in enumerate(queries, 1))
        return QueryPlan(analysis.normalized, steps)
