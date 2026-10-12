class PromptBuilder:
    """Construye contexto para aclaraciones futuras; no invoca LLMs."""
    def clarification(self, proposal):
        if not proposal.requires_input and not proposal.issues: return None
        lines=['La propuesta requiere revisión antes de ejecutarse.']
        if proposal.requires_input: lines.append('Parámetros faltantes: '+', '.join(proposal.requires_input))
        lines.extend('- '+issue.message for issue in proposal.issues)
        return '\n'.join(lines)
    def review_prompt(self, proposal):
        body='\n'.join(f'{step.id}: {step.command_template or "sin comando"}' for step in proposal.steps)
        return f'Revisá esta propuesta sin ejecutarla:\n{body}'
