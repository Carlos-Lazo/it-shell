import json
class ResponseFormatterService:
    def format(self, response, output_format):
        if output_format=='json': return json.dumps(response.to_dict(), ensure_ascii=False, indent=2)
        if output_format=='markdown':
            lines=[f'# Propuesta: {response.plan.goal}', '']
            for step in response.proposal.steps: lines += [f'## {step.id}', f'`{step.command_template or "Sin comando candidato"}`', '']
            return '\n'.join(lines)
        if output_format=='cli':
            lines=[f'Propuesta: {response.plan.goal}']
            lines += [f'{step.id}. {step.command_template or "Sin comando candidato"}' for step in response.proposal.steps]
            return '\n'.join(lines)
        raise ValueError(f'Formato no soportado: {output_format}')
