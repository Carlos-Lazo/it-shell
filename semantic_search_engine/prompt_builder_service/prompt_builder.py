import re
from dataclasses import dataclass

from semantic_search_engine.shared.contracts import PipelineProposal


@dataclass(frozen=True)
class ExplanationStep:
    step_id: str
    intent: str
    command: str | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationContext:
    goal: str
    steps: tuple[ExplanationStep, ...]
    warnings: tuple[str, ...]
    missing_input: tuple[str, ...]


class PromptBuilder:
    """Construye explicaciones y contexto restringido sin modificar comandos."""

    FLAG_PATTERN = re.compile(r'(?<!\S)--?[A-Za-z][A-Za-z0-9-]*')

    def clarification(self, proposal: PipelineProposal) -> str | None:
        if not proposal.requires_input and not proposal.issues:
            return None
        lines = ['La propuesta requiere revisión antes de ejecutarse.']
        if proposal.requires_input:
            lines.append('Información necesaria: ' + ', '.join(proposal.requires_input))
        for issue in proposal.issues:
            lines.append('- ' + issue.message)
            if issue.recommendation:
                lines.append('  Recomendación: ' + issue.recommendation)
        return '\n'.join(lines)

    def explanation_context(self, proposal: PipelineProposal) -> ExplanationContext:
        steps = tuple(
            ExplanationStep(
                step.id,
                step.selected.intent if step.selected else step.query,
                step.command_template,
                tuple(self.FLAG_PATTERN.findall(step.command_template or '')),
            )
            for step in proposal.steps
        )
        warnings = tuple(
            f'{issue.level.upper()}: {issue.message}'
            + (f' Recomendación: {issue.recommendation}' if issue.recommendation else '')
            for issue in proposal.issues
        )
        return ExplanationContext(
            proposal.goal,
            steps,
            warnings,
            proposal.requires_input,
        )

    def explanation_prompt(self, proposal: PipelineProposal) -> str:
        context = self.explanation_context(proposal)
        lines = [
            'Explica la propuesta de comandos de forma breve y pedagógica.',
            'Reglas obligatorias:',
            '- No cambies, completes ni reordenes ningún comando.',
            '- No inventes opciones, argumentos ni efectos que no estén en el contexto.',
            '- No indiques que un comando es seguro si hay avisos asociados.',
            f'Objetivo del usuario: {context.goal}',
        ]
        for step in context.steps:
            command = step.command or 'sin comando recomendado'
            flags = ', '.join(step.flags) if step.flags else 'ninguna detectada'
            lines.append(
                f'{step.step_id}: intención={step.intent}; comando exacto={command}; opciones={flags}'
            )
        for warning in context.warnings:
            lines.append(f'Aviso: {warning}')
        if context.missing_input:
            lines.append('Información faltante: ' + ', '.join(context.missing_input))
        return '\n'.join(lines)

    def offline_explanation(self, proposal: PipelineProposal) -> str:
        context = self.explanation_context(proposal)
        lines = [f'Objetivo: {context.goal}']
        for step in context.steps:
            if step.command:
                lines.append(f'{step.step_id}: {step.intent}. Comando sugerido: {step.command}')
            else:
                lines.append(f'{step.step_id}: {step.intent}. No hay un comando confiable seleccionado.')
        if context.missing_input:
            lines.append('Antes de continuar, falta: ' + ', '.join(context.missing_input) + '.')
        if context.warnings:
            lines.append('Advertencias: ' + ' '.join(context.warnings))
        return '\n'.join(lines)

    def review_prompt(self, proposal: PipelineProposal) -> str:
        return self.explanation_prompt(proposal)
