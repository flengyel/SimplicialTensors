"""Check that repository Markdown and notebook links survive directory moves."""

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r'!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)')
REFERENCE_LINK = re.compile(r'^\s{0,3}\[[^\]^]+\]:\s*(<[^>]+>|\S+)', re.MULTILINE)


def _prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token[0]
            elif token[0] == fence:
                fence = None
            continue
        if fence is None:
            lines.append(re.sub(r'`+[^`]*`+', '', line))
    return '\n'.join(lines)


def _targets(text):
    prose = _prose(text)
    for regex in (INLINE_LINK, REFERENCE_LINK):
        for match in regex.finditer(prose):
            target = match.group(1).strip('<>')
            parsed = urlsplit(target)
            if not parsed.scheme and not parsed.netloc and parsed.path:
                yield unquote(parsed.path)


def test_local_documentation_links_resolve():
    documents = list(ROOT.glob('*.md'))
    for directory in ('docs', 'experiments', 'tex'):
        documents.extend((ROOT / directory).rglob('*.md'))
    missing = []
    for path in documents:
        for target in _targets(path.read_text(encoding='utf-8')):
            if not (path.parent / target).exists():
                missing.append(f'{path.relative_to(ROOT)}: {target}')
    for path in (ROOT / 'experiments/notebooks').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                for target in _targets(''.join(cell['source'])):
                    if not (path.parent / target).exists():
                        missing.append(f'{path.relative_to(ROOT)}: {target}')
    assert not missing, 'Missing local link targets:\n' + '\n'.join(sorted(missing))
