import re
from semantic_search.shared.contracts import PipelineProposal, ValidationIssue

class ProposalValidator:
    DANGEROUS=(r'\brm\s+-rf',r'\bdd\b',r'\bmkfs\b',r'\bshutdown\b',r'\breboot\b')
    def validate(self, proposal):
        issues=list(proposal.issues); missing=set()
        for step in proposal.steps:
            if step.selected is None:
                issues.append(ValidationIssue('warning','no_candidate','No se encontró un candidato confiable',step.id)); continue
            template=step.command_template or ''
            missing.update(re.findall(r'<([^>]+)>', template))
            if any(re.search(pattern, template) for pattern in self.DANGEROUS):
                issues.append(ValidationIssue('warning','dangerous_command','La propuesta requiere revisión humana',step.id))
        return PipelineProposal(proposal.goal, proposal.steps, tuple(sorted(missing)), tuple(issues), executable=False)
