class PipelineBuilderService:
    def __init__(self, resolver, builder, validator): self.resolver, self.builder, self.validator=resolver, builder, validator
    def propose(self, plan, retrieved, reranker_service):
        ranked={}
        for step in plan.steps:
            candidates=[self.resolver.resolve(alias, score) for alias, score in retrieved[step.id]]
            ranked[step.id]=reranker_service.rerank(step.query, candidates)
        return self.validator.validate(self.builder.build(plan, ranked))
