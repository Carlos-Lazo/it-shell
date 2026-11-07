import re
from dataclasses import replace
from semantic_search_engine.retrieval_service.planner import _ambiguous_argument_types, _extract_arguments
from semantic_search_engine.shared.contracts import PipelineProposal, PipelineStep, ValidationIssue

class ProposalValidator:
    PLACEHOLDER_PATTERN = re.compile(r'<([^<>]+)>|\{([^{}]+)\}')
    ROOT_RECURSIVE_DELETE_REQUEST = re.compile(
        r'\brm\b'
        r'(?=[^;&|\n]*\s(?:--recursive|-[a-z]*r[a-z]*)(?:[\s,]|$))'
        r'(?=[^;&|\n]*\s(?:--force|-[a-z]*f[a-z]*)(?:[\s,]|$))'
        r'[^;&|\n]*\s/(?:[\s,.;]|$)',
        re.IGNORECASE,
    )
    COMPRESSION_PATTERN = re.compile(
        r'\b(?:compress(?:es|ed|ing|ion)?|comprimir|comprime|comprim(?:ir|iendo|ido|ida))\b',
        re.IGNORECASE,
    )
    GENERIC_FILE_PATTERN = re.compile(r'\b(?:file|files|archivo|archivos)\b', re.IGNORECASE)
    DESTINATION_PATTERN = re.compile(
        r'\b(?:to|into|as|hacia|en)\s+(?:"[^"]+"|\'[^\']+\'|(?:~|\.{1,2})?/|[\w.-]+\.[\w.-]+)',
        re.IGNORECASE,
    )

    DESTRUCTIVE_RULES = (
        (
            'critical',
            'destructive_command',
            re.compile(
                r'\brm\b(?=[^;&|\n]*\s(?:--recursive|-[a-zA-Z]*r[a-zA-Z]*)(?:[\s,]|$))',
                re.IGNORECASE,
            ),
            'Eliminación recursiva de archivos y directorios.',
            'No ejecutes el comando sin verificar cada ruta y contar con una copia de seguridad.',
        ),
        (
            'critical',
            'destructive_command',
            re.compile(r'\b(?:shred|wipefs|mkfs(?:\.[a-zA-Z0-9_-]+)?|fdisk|parted|gdisk)\b', re.IGNORECASE),
            'Comando que puede destruir datos, firmas de sistemas de archivos o particiones.',
            'Verifica el dispositivo objetivo y utiliza un entorno de prueba antes de cualquier operación.',
        ),
        (
            'critical',
            'destructive_command',
            re.compile(r'\bdd\b(?=\s|$)', re.IGNORECASE),
            'Escritura de bajo nivel que puede sobrescribir dispositivos o archivos.',
            'Confirma origen y destino; no ejecutes hasta validar que el destino no contiene datos necesarios.',
        ),
        (
            'critical',
            'destructive_command',
            re.compile(r'\btruncate\b(?=\s|$)', re.IGNORECASE),
            'El comando truncate puede reducir un archivo y descartar datos.',
            'Comprueba el tamaño y la ruta destino y conserva una copia antes de modificar el archivo.',
        ),
        (
            'critical',
            'critical_redirection',
            re.compile(r'>\s*[\'"]?/(?:etc|boot)(?:/|[\'"]?(?:\s|$))', re.IGNORECASE),
            'Redirección de escritura hacia una ruta crítica del sistema (/etc o /boot).',
            'No sobrescribas archivos del sistema; redirige a una ruta temporal bajo tu control.',
        ),
        (
            'critical',
            'destructive_command',
            re.compile(r'>\s*[\'"]?/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|loop\d*)(?:p?\d+)?', re.IGNORECASE),
            'Redirección directa a un dispositivo de bloque.',
            'No escribas directamente a dispositivos; valida el destino y usa una herramienta segura.',
        ),
        (
            'critical',
            'fork_bomb',
            re.compile(r':\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&?\s*\}\s*;?\s*:', re.IGNORECASE),
            'Patrón de fork bomb que puede agotar los recursos del sistema.',
            'No ejecutes este patrón. Elimínalo y usa una prueba controlada con límites de recursos.',
        ),
    )

    PRIVILEGED_RULES = (
        (
            'warning',
            'privileged_command',
            re.compile(r'\bsudo\b(?!\s+-l(?:\s|$))|\bsu\b', re.IGNORECASE),
            'Ejecución con privilegios elevados.',
            'Evita privilegios elevados si no son imprescindibles y revisa exactamente qué acción se realizará.',
        ),
        (
            'warning',
            'privileged_command',
            re.compile(r'\b(?:shutdown|reboot|poweroff|halt)\b|\binit\s+[06]\b', re.IGNORECASE),
            'Apagado, reinicio o cambio del estado de ejecución del sistema.',
            'Confirma el impacto sobre usuarios y servicios antes de solicitar esta operación.',
        ),
        (
            'warning',
            'privileged_command',
            re.compile(
                r'\bchmod\s+(?:(?:-[a-zA-Z]*R|--recursive)\s+)?(?:0?777|a\+rwx)\b'
                r'|\bchmod\s+(?:-[a-zA-Z]*R|--recursive)\b',
                re.IGNORECASE,
            ),
            'Asignación de permisos excesivamente abiertos.',
            'Aplica el mínimo de permisos necesario y evita conceder escritura a todos los usuarios.',
        ),
        (
            'warning',
            'privileged_command',
            re.compile(
                r'\bchown\s+(?:(?:-[a-zA-Z]*R|--recursive)\s+\S+|root(?::root)?\b)',
                re.IGNORECASE,
            ),
            'Cambio recursivo de permisos o asignación de propiedad a root.',
            'Verifica el alcance del cambio y limita la operación a rutas y usuarios necesarios.',
        ),
        (
            'warning',
            'privileged_command',
            re.compile(r'\b(?:iptables\s+-[FX]|ufw\s+disable)\b|\bkill(?:all)?\s+(?:-9|-KILL|-s\s+KILL)\b', re.IGNORECASE),
            'Alteración del firewall o terminación forzada de procesos.',
            'Revisa el servicio afectado y prefiere una operación reversible y específica.',
        ),
    )

    AVAILABILITY_RULES = (
        (
            'warning',
            'resource_exhaustion',
            re.compile(r'\bwhile\s+(?:true|:)\s*;\s*do\b|\bfor\s*\(\s*;\s*;\s*\)', re.IGNORECASE),
            'Bucle sin condición de terminación visible que puede consumir recursos.',
            'Añade una condición de salida y límites de tiempo o recursos antes de ejecutarlo.',
        ),
    )

    @classmethod
    def _missing_inputs(cls, step: PipelineStep) -> tuple[str, ...]:
        missing = {
            (match.group(1) or match.group(2)).strip().lower().replace('-', '_').replace(' ', '_')
            for match in cls.PLACEHOLDER_PATTERN.finditer(step.command_template or '')
            if match.group(1) or '|' not in match.group(2)
        }
        argument_types = {kind for kind, _ in _extract_arguments(step.query)}
        ambiguous_types = set(step.ambiguous_arguments or _ambiguous_argument_types(step.query))
        missing.update(ambiguous_types)

        context = f'{step.query} {step.selected.intent if step.selected else ""}'
        if cls.COMPRESSION_PATTERN.search(context):
            if cls.GENERIC_FILE_PATTERN.search(step.query) and not argument_types.intersection({'path', 'filename'}):
                missing.add('filename')
            if not step.redirect_target and not cls.DESTINATION_PATTERN.search(step.query):
                missing.add('destination')
        if step.redirect_operator and not step.redirect_target:
            missing.add('destination')

        return tuple(sorted(missing))

    def validate(self, proposal: PipelineProposal) -> PipelineProposal:
        issues = list(proposal.issues)
        missing = set(proposal.requires_input)
        validated_steps = []

        for original_step in proposal.steps:
            root_delete_requested = bool(
                self.ROOT_RECURSIVE_DELETE_REQUEST.search(original_step.query)
            )
            step = (
                replace(
                    original_step,
                    selected=None,
                    alternatives=(),
                    command_template=None,
                )
                if root_delete_requested
                else original_step
            )
            validated_steps.append(step)

            if root_delete_requested:
                issues.append(ValidationIssue(
                    'critical',
                    'destructive_command',
                    'La solicitud incluye una eliminación recursiva forzada de la raíz del sistema (/).',
                    step.id,
                    'Es extremadamente peligroso: no se debe recomendar ni ejecutar. Puedo explicar el riesgo sin proporcionar ese comando.',
                ))
            elif step.selected is None:
                issues.append(ValidationIssue(
                    'warning',
                    'no_candidate',
                    'No se encontró un candidato confiable.',
                    step.id,
                    'Reformula la consulta o proporciona más contexto antes de elegir un comando.',
                ))
            template = step.command_template or ''
            if step.selected is not None:
                step_missing = self._missing_inputs(step)
                missing.update(step_missing)
                if step_missing:
                    labels = ', '.join(step_missing)
                    issues.append(ValidationIssue(
                        'warning',
                        'missing_input',
                        f'Falta información para {step.id}: especifica {labels}.',
                        step.id,
                        'Completa los datos solicitados; no se deben inferir ni inventar.',
                    ))

            checked_text = f'{template} {step.redirect_operator or ""} {step.redirect_target or ""}'
            for level, code, pattern, message, recommendation in (
                self.DESTRUCTIVE_RULES + self.PRIVILEGED_RULES + self.AVAILABILITY_RULES
            ):
                if pattern.search(checked_text):
                    issues.append(ValidationIssue(
                        level,
                        code,
                        message,
                        step.id,
                        recommendation,
                    ))

        return PipelineProposal(
            goal=proposal.goal,
            steps=tuple(validated_steps),
            requires_input=tuple(sorted(missing)),
            issues=tuple(issues),
            executable=False,
        )
