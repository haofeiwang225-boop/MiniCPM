"""Read-only split provenance and source sidecars; never repartition examples."""
import hashlib
import json
from pathlib import Path
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_version():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')))


def build_source_metadata(data_path, source_path):
    rows = json.loads(Path(data_path).read_text())
    wanted = {}
    for index, row in enumerate(rows):
        messages = row['messages']
        if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
            raise ValueError('Source matching requires the existing single-turn data format')
        wanted.setdefault((messages[0]['content'], messages[1]['content']), []).append(index)
    matches = {}
    with Path(source_path).open() as stream:
        for line_index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get('query'), row.get('response'))
            if key not in wanted:
                continue
            family = row.get('type', '').split('_')[0]
            original = row.get('original_question')
            if family not in ('GSM', 'MATH') or not isinstance(original, str):
                raise ValueError(f'Missing source family/group at source line {line_index + 1}')
            group = hashlib.sha256(original.encode()).hexdigest()
            info = dict(family=family, source_type=row['type'], original_question_sha256=group,
                        source_line_1based=line_index + 1)
            if key in matches and any(matches[key][k] != info[k] for k in ('family', 'original_question_sha256')):
                raise ValueError('Ambiguous source family/group for identical sample')
            matches.setdefault(key, info)
    result = [None] * len(rows)
    for key, indices in wanted.items():
        if key not in matches:
            raise ValueError(f'No exact query/response source match for sample {indices[0]}')
        for index in indices:
            row_hash = hashlib.sha256(json.dumps(rows[index], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            result[index] = dict(index=index, sample_sha256=row_hash, **matches[key])
    return dict(data_sha256=file_sha256(data_path), source_path=str(Path(source_path).resolve()),
                source_sha256=file_sha256(source_path), num_samples=len(rows), samples=result)


def save_source_metadata(data_path, source_path, output_path):
    metadata = build_source_metadata(data_path, source_path)
    output_path = Path(output_path)
    if output_path.exists():
        if json.loads(output_path.read_text()) != metadata:
            raise ValueError(f'Refusing to overwrite different metadata: {output_path}')
    else:
        with output_path.open('x') as stream:
            json.dump(metadata, stream, indent=2)
    return metadata
