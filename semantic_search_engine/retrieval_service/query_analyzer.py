import re
from semantic_search.shared.contracts import QueryAnalysis

class QueryAnalyzer:
    COMPLEX_MARKERS=(' y luego ', ' después ', ' luego ', ' and then ', ' then ')
    SPANISH_MARKERS=('buscar','archivo','comprimir','enviar','mostrar','crear','eliminar')
    def analyze(self, query: str) -> QueryAnalysis:
        normalized=re.sub(r'\s+', ' ', query.strip())
        lower=f' {normalized.lower()} '
        return QueryAnalysis(query, normalized, 'es' if any(word in lower for word in self.SPANISH_MARKERS) else 'en', any(mark in lower for mark in self.COMPLEX_MARKERS))
