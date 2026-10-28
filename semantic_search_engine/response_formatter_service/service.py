import json

class ResponseFormatterService:
    def format(self, response, output_format):
        if output_format == 'json':
            return json.dumps(response.to_dict(), ensure_ascii=False, indent=2)
        if output_format == 'markdown':
            lines = [f'# Propuesta: {response.plan.goal}', '']
            for step in response.proposal.steps:
                cmd = f'`{step.command_template}`' if step.selected else '*No se encontró una recomendación confiable*'
                lines += [f'## {step.id}', cmd, '']
            if response.proposal.issues:
                lines += ['### Avisos', '']
                for issue in response.proposal.issues:
                    lines.append(f'- **{issue.level.upper()}**: {issue.message}')
                    if issue.recommendation:
                        lines.append(f'  - Recomendación: {issue.recommendation}')
                lines.append('')
            return '\n'.join(lines)
        if output_format == 'cli':
            lines = [f'Propuesta: {response.plan.goal}']
            for step in response.proposal.steps:
                cmd = step.command_template if step.selected else 'No se encontró una recomendación confiable'
                lines.append(f'{step.id}. {cmd}')
            if response.proposal.issues:
                lines.append('Avisos:')
                for issue in response.proposal.issues:
                    lines.append(f'  - [{issue.level.upper()}] {issue.message}')
                    if issue.recommendation:
                        lines.append(f'    Recomendación: {issue.recommendation}')
            return '\n'.join(lines)
        raise ValueError(f'Formato no soportado: {output_format}')
