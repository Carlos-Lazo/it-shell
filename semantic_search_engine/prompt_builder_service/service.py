class PromptBuilderService:
    def __init__(self, builder): self.builder=builder
    def clarification(self, proposal): return self.builder.clarification(proposal)
