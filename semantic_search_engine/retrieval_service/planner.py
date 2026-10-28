import re
from collections import defaultdict
from shlex import quote, split as shell_split

from semantic_search_engine.shared.contracts import PlanStep, QueryAnalysis, QueryPlan

SEQUENCE_SPLITTER = re.compile(
    r'\s*(?:&&|\|\||\||;|>>?)\s*'
    r'|'
    r'\s+(?:'
    r'(?:and\s+)?(?:then|next|after\s+that|afterwards|followed\s+by)'
    r'|'
    r'(?:y\s+)?(?:luego|después|despues|a\s+continuación|a\s+continuacion|seguido\s+de)'
    r')\s+',
    re.IGNORECASE,
)
NATURAL_SEQUENCE_SPLITTER = re.compile(
    r'\s+((?:and\s+)?(?:then|next|after\s+that|afterwards|followed\s+by)'
    r'|(?:y\s+)?(?:luego|después|despues|a\s+continuación|a\s+continuacion|seguido\s+de))\s+',
    re.IGNORECASE,
)
SHELL_CONNECTORS = ('&&', '||', '|', ';')

ARGUMENT_PATTERNS = (
    (
        'extension',
        re.compile(r'(?<![\w.])\*\.[A-Za-z0-9_*?]+(?:\.[A-Za-z0-9_*?]+)*'),
    ),
    (
        'size',
        re.compile(r'(?<![\w.])[+-]\d+(?:\.\d+)?[KMGTPE]?B?(?!\w)', re.IGNORECASE),
    ),
    (
        'path',
        re.compile(
            r'''(?<![\w])(?:"((?:~|\.{1,2})?/[^"]+)"|'((?:~|\.{1,2})?/[^']+)'|((?:~|\.{1,2})?/[^\s,;|&]+))'''
        ),
    ),
    (
        'filename',
        re.compile(r'(?<![\w./])[\w-][\w.-]*\.[A-Za-z0-9][A-Za-z0-9._-]*'),
    ),
)

PLACEHOLDER_TYPES = {
    'dir': 'path',
    'directory': 'path',
    'dir_path': 'path',
    'path': 'path',
    'output_dir': 'path',
    'file': 'filename',
    'filename': 'filename',
    'archive': 'filename',
    'extension': 'extension',
    'ext': 'extension',
    'size': 'size',
    'limit': 'size',
    'threshold': 'size',
}


def _trim_argument(value: str) -> str:
    return value.rstrip('.,!?)]}\'"')


def _extract_arguments(query: str) -> tuple[tuple[str, str], ...]:
    matches: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for argument_type, pattern in ARGUMENT_PATTERNS:
        for match in pattern.finditer(query):
            value = next((group for group in match.groups() if group), '') if argument_type == 'path' else match.group()
            value = _trim_argument(value)
            if value:
                matches[argument_type].append((match.start(), value))

    return tuple(
        (argument_type, values[0][1])
        for argument_type, values in sorted(
            matches.items(), key=lambda item: min(position for position, _ in item[1])
        )
        if len({value for _, value in values}) == 1
    )


def _ambiguous_argument_types(query: str) -> tuple[str, ...]:
    matches: dict[str, set[str]] = defaultdict(set)
    for argument_type, pattern in ARGUMENT_PATTERNS:
        for match in pattern.finditer(query):
            value = next((group for group in match.groups() if group), '') if argument_type == 'path' else match.group()
            value = _trim_argument(value)
            if value:
                matches[argument_type].add(value)
    return tuple(sorted(kind for kind, values in matches.items() if len(values) > 1))


def _fill_placeholders(
    template: str,
    arguments: tuple[tuple[str, str], ...],
    redirect_target: str | None = None,
) -> str:
    values = dict(arguments)

    def replace(match: re.Match[str]) -> str:
        placeholder = (match.group(1) or match.group(2)).strip().lower().replace('-', '_').replace(' ', '_')
        if placeholder in {'destination', 'output', 'output_file', 'target'} and redirect_target:
            return quote(redirect_target)
        argument_type = PLACEHOLDER_TYPES.get(placeholder)
        value = values.get(argument_type) if argument_type else None
        if argument_type == 'filename' and value is None:
            path = values.get('path')
            if path and re.search(r'\.[^/]+$', path):
                value = path
        if value is None:
            return match.group()
        return quote(value)

    return re.sub(r'<([^<>]+)>|\{([^{}]+)\}', replace, template)


def _split_shell_connectors(query: str) -> list[tuple[str, str | None]]:
    segments: list[tuple[str, str | None]] = []
    start = 0
    connector: str | None = None
    quote_char: str | None = None
    escaped = False
    index = 0

    while index < len(query):
        char = query[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == '\\' and quote_char != "'":
            escaped = True
            index += 1
            continue
        if quote_char:
            if char == quote_char:
                quote_char = None
            index += 1
            continue
        if char in ("'", '"'):
            quote_char = char
            index += 1
            continue

        operator = next(
            (candidate for candidate in ('&&', '||', '|', ';') if query.startswith(candidate, index)),
            None,
        )
        if operator:
            part = query[start:index].strip()
            if part:
                segments.append((part, connector))
            connector = operator
            index += len(operator)
            start = index
            continue
        index += 1

    part = query[start:].strip()
    if part:
        segments.append((part, connector))
    return segments


def _extract_redirection(query: str) -> tuple[str, str | None, str | None]:
    quote_char: str | None = None
    escaped = False
    index = 0

    while index < len(query):
        char = query[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == '\\' and quote_char != "'":
            escaped = True
            index += 1
            continue
        if quote_char:
            if char == quote_char:
                quote_char = None
            index += 1
            continue
        if char in ("'", '"'):
            quote_char = char
            index += 1
            continue
        if char == '>':
            operator = '>>' if query.startswith('>>', index) else '>'
            target = query[index + len(operator):].strip()
            parsed_target = shell_split(target) if target else []
            normalized_target = parsed_target[0] if len(parsed_target) == 1 else None
            return query[:index].strip(), operator, normalized_target
        index += 1

    return query.strip(), None, None


class RulePlanner:
    """Divide secuencias explícitas sin inventar comandos ni argumentos."""
    splitter = SEQUENCE_SPLITTER

    def plan(self, analysis: QueryAnalysis) -> QueryPlan:
        parsed_steps: list[tuple[str, str | None, str | None, str | None]] = []
        for shell_part, shell_connector in _split_shell_connectors(analysis.normalized):
            natural_parts = NATURAL_SEQUENCE_SPLITTER.split(shell_part)
            for index in range(0, len(natural_parts), 2):
                query = natural_parts[index].strip(' ,.;')
                if not query:
                    continue
                connector = shell_connector if index == 0 else 'natural'
                query, redirect_operator, redirect_target = _extract_redirection(query)
                query = query.strip(' ,.;')
                if query:
                    parsed_steps.append((query, connector, redirect_operator, redirect_target))

        if not parsed_steps:
            parsed_steps.append((analysis.normalized, None, None, None))

        steps = tuple(
            PlanStep(
                f'step_{i}',
                query,
                (f'step_{i-1}',) if i > 1 else (),
                _extract_arguments(query),
                connector,
                redirect_operator,
                redirect_target,
                _ambiguous_argument_types(query),
            )
            for i, (query, connector, redirect_operator, redirect_target) in enumerate(parsed_steps, 1)
        )
        return QueryPlan(analysis.normalized, steps)
