from semantic_search_engine.shared.contracts import PipelineProposal, PipelineStep

class PipelineBuilder:
    def __init__(self, selection): self.selection=selection
    def choose(self, ranked):
        if not ranked or ranked[0].retrieval_score < self.selection.min_score:
            return None, ()
        primary = ranked[0]
        alternatives = []
        for candidate in ranked[1:]:
            if candidate.retrieval_score < self.selection.min_score:
                break
            if primary.reranker_score is None and (primary.score - candidate.score > self.selection.alternative_margin):
                break
            alternatives.append(candidate)
            if len(alternatives) == self.selection.max_alternatives - 1:
                break
        return primary, tuple(alternatives)
    def build(self, plan, ranked_by_step):
        steps=[]
        for step in plan.steps:
            selected, alternatives=self.choose(ranked_by_step.get(step.id, []))
            template = (selected.example or selected.command) if selected else None
            steps.append(PipelineStep(step.id, step.query, step.depends_on, selected, alternatives, template))
        return PipelineProposal(plan.goal, tuple(steps), executable=False)
