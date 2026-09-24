"""Derive a content-free comparison outside a sealed session, entirely offline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from recorded_session import digest, verify_evidence
from session_history import inspect_session
from session_comparison import build_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--backend-rows', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    directory, output = args.session.resolve(), args.output.resolve()
    if output.is_relative_to(directory):
        parser.error('derived output must be outside the sealed session')
    if output.exists():
        parser.error('output already exists')
    try:
        inspected = inspect_session(directory)
        verified = verify_evidence(directory)
        if not inspected['complete'] or not verified['complete']:
            raise ValueError('session inventory or journal integrity failure')
        inputs = {}

        def load(path, label, jsonl=False):
            path = path.resolve()
            inputs[label] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            text = path.read_text(encoding='utf-8-sig')
            return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)

        summary = load(directory / 'summary.json', 'session_summary')
        summary['journal_complete'] = inspected['complete']
        load(directory / 'events.jsonl', 'journal', jsonl=True)
        load(directory / 'manifest.json', 'manifest')
        load(directory / 'evidence-inventory.json', 'inventory')
        rows = load(args.backend_rows or directory / 'wire-requests.json', 'wire_rows')
        if not isinstance(rows, list):
            raise ValueError('wire rows must be a list')
        observations = []
        root = args.observations.resolve() if args.observations else directory
        if not root.is_dir():
            raise ValueError('observation directory missing')
        for index, path in enumerate(sorted(root.rglob('timing.json'))):
            observation = load(path, f'observation_{index}')
            observation['_observation_refs'] = [path.relative_to(root).as_posix()]
            if path.resolve().is_relative_to(directory):
                observation['_observation_refs'].append(path.resolve().relative_to(directory).as_posix())
            # Check hashes against actual serialized bytes when they are present.
            request = path.with_name('request.bin')
            if request.is_file():
                actual = digest(request)
                inputs[f'observation_request_{index}'] = {'sha256': actual, 'bytes': request.stat().st_size}
                if observation.get('request_sha256') not in (None, actual):
                    raise ValueError('observation request bytes hash mismatch')
                observation['request_sha256'] = actual
            observations.append(observation)
        journal_path = directory / 'visible-test-evidence' / 'visible-tests.jsonl'
        tests = load(journal_path, 'test_journal', jsonl=True) if journal_path.is_file() else None
        report = build_comparison(rows, inspected['events'], observations, summary, tests)
        report['inputs'] = inputs
        report['input_role'] = 'backend_rows' if args.backend_rows else 'session_wire_rows'
        rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        with output.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        # Never echo private provider data or exception strings to public output.
        parser.error('comparison failed: ' + type(error).__name__)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
