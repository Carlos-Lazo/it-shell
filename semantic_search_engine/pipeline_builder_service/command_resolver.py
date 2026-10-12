from pathlib import Path
import yaml
from semantic_search_engine.shared.contracts import CandidateCommand

class CommandResolver:
    def __init__(self, corpus_path: Path): self.corpus_path, self.cache=corpus_path, {}
    def resolve(self, alias: str, retrieval_score: float) -> CandidateCommand:
        category, raw_id=alias.rsplit(':', 1)
        source_id=int(raw_id)
        if category not in self.cache:
            self.cache[category]=yaml.safe_load((self.corpus_path/f'{category}.yaml').read_text(encoding='utf-8'))
        row=next((item for item in self.cache[category]['documents'] if item['id']==source_id), None)
        if row is None: raise KeyError(f'Alias inexistente: {alias}')
        return CandidateCommand(alias, category, source_id, row.get('intent',''), row.get('command',''), row.get('example',''), retrieval_score)
