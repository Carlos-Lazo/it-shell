import re
from semantic_search_engine.shared.contracts import PipelineProposal, ValidationIssue

class ProposalValidator:
    # Comandos destructivos o con potencial pérdida irreversible de datos
    DESTRUCTIVE_PATTERNS = (
        (r'\brm\s+-(?:[a-zA-Z]*r[a-zA-Z]*f|[a-zA-Z]*f[a-zA-Z]*r)\b', 'Eliminación recursiva forzada de archivos (rm -rf)'),
        (r'\bshred\b', 'Sobreescritura destructiva de archivos (shred)'),
        (r'\bwipefs\b', 'Borrado de firmas de sistemas de archivos (wipefs)'),
        (r'\bmkfs(?:\.[a-zA-Z0-9_-]+)?\b', 'Creación o formateo de sistemas de archivos (mkfs)'),
        (r'\b(?:fdisk|parted|gdisk)\b', 'Modificación de tablas de particiones de disco'),
        (r'\bdd\b', 'Escritura de bajo nivel en dispositivos o particiones (dd)'),
        (r'>\s*/dev/(?:sd[a-z]|nvme[0-9]|vd[a-z]|loop)', 'Redirección destructiva directa a dispositivo de bloque'),
    )

    # Comandos con privilegios elevados o alteración de estado del sistema/red
    PRIVILEGED_PATTERNS = (
        (r'\b(?:sudo|su)\b', 'Ejecución con privilegios de superusuario (root)'),
        (r'\b(?:shutdown|reboot|poweroff|halt)\b', 'Apagado o reinicio del sistema'),
        (r'\binit\s+[06]\b', 'Cambio de nivel de ejecución del sistema'),
        (r'\bchmod\s+(?:-[a-zA-Z]*R|--recursive)\b', 'Cambio recursivo de permisos en el sistema de archivos'),
        (r'\bchmod\s+(?:0?[67]77)\b', 'Asignación de permisos abiertos indiscriminados (777)'),
        (r'\bchown\s+(?:-[a-zA-Z]*R|--recursive)\b', 'Cambio recursivo de propietario de archivos'),
        (r'\b(?:iptables\s+-[FX]|ufw\s+disable)\b', 'Alteración o desactivación de reglas de firewall'),
        (r'\bkill(?:all)?\s+(?:-9|-KILL|-s\s+KILL)\b', 'Terminación forzada e inmediata de procesos (SIGKILL)'),
    )

    def validate(self, proposal: PipelineProposal) -> PipelineProposal:
        issues = list(proposal.issues)
        missing = set()

        for step in proposal.steps:
            if step.selected is None:
                issues.append(ValidationIssue('warning', 'no_candidate', 'No se encontró un candidato confiable', step.id))
                continue

            template = step.command_template or ''
            missing.update(re.findall(r'<([^>]+)>', template))

            for pattern, desc in self.DESTRUCTIVE_PATTERNS:
                if re.search(pattern, template, re.IGNORECASE):
                    issues.append(ValidationIssue('warning', 'destructive_command', f'Comando destructivo: {desc}', step.id))
                    break

            for pattern, desc in self.PRIVILEGED_PATTERNS:
                if re.search(pattern, template, re.IGNORECASE):
                    issues.append(ValidationIssue('warning', 'privileged_command', f'Comando privilegiado: {desc}', step.id))
                    break

        return PipelineProposal(
            goal=proposal.goal,
            steps=proposal.steps,
            requires_input=tuple(sorted(missing)),
            issues=tuple(issues),
            executable=False,
        )
