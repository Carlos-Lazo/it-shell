class RetrievalService:
    def __init__(self, analyzer, planner, retriever): self.analyzer, self.planner, self.retriever=analyzer, planner, retriever
    def retrieve_plan(self, query, top_k):
        analysis=self.analyzer.analyze(query)
        plan=self.planner.plan(analysis)
        return analysis, plan, {step.id: self.retriever.retrieve(step.query, top_k) for step in plan.steps}
