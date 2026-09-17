import re
from semantic_search_engine.shared.contracts import QueryAnalysis
from semantic_search_engine.retrieval_service.planner import SEQUENCE_SPLITTER

class QueryAnalyzer:
    SPANISH_MARKERS = ('buscar', 'archivo', 'comprimir', 'enviar', 'mostrar', 'crear', 'eliminar')

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = re.sub(r'\s+', ' ', query.strip())
        lower = f' {normalized.lower()} '
        is_spanish = any(word in lower for word in self.SPANISH_MARKERS)
        is_complex = bool(SEQUENCE_SPLITTER.search(normalized))
        return QueryAnalysis(query, normalized, 'es' if is_spanish else 'en', is_complex)
