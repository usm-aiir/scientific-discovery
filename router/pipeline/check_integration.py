"""Check that connecting the router to table search still gives the same results."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from router_src.data import evaluation_coverage, file_hash, load_records, write_json
from .search import DEFAULT_CONFIG, Orchestrator, load_config


def equivalent(left, right):
    """Check that two results match, allowing tiny differences in their scores."""
    if isinstance(left, float) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def stable_evidence(result):
    """Leave out dates and times so we can compare the actual results."""
    return {key: value for key, value in result.items() if key not in ('created_at_utc', 'timings')}


def main(argv=None):
    """Try a few validation questions and compare the two ways of running table search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--validation-file', type=Path, default=DEFAULT_CONFIG.parent.parent / 'data' / 'Val.json')
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--output', type=Path, help='New directory; existing results are refused')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('count must be positive')
    output = args.output or (DEFAULT_CONFIG.parent.parent / 'results') / (
        'integration_smoke_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        parser.error('Output directory exists; choose a fresh path')
    config = load_config(args.config)
    if 'rewriter' in config:
        parser.error('Integration check requires pipeline/settings.json; '
                     'use python -m rewrite_src.evaluate --limit 3 for Gemma rewriting')
    coverage, manifest_hash = evaluation_coverage(args.validation_file, 'Val')
    data_hash = file_hash(args.validation_file)
    records = load_records(args.validation_file, 'Val')
    # Choose the questions before seeing any model predictions.
    selected = [r for r in records if ['table'] in r['acceptable_sets']][:args.count]
    if len(selected) != args.count:
        parser.error('Not enough validation records with a known table-only acceptable route')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'questions.json', selected)
    protocol = {
        'split': 'Val', 'validation_file': str(args.validation_file.resolve()),
        'validation_sha256': data_hash, 'manifest_sha256': manifest_hash,
        'review_coverage': coverage, 'configuration': config,
        'runtime_config_sha256': file_hash(args.config),
        'selection': 'first N records in original Val order with table singleton in acceptable_sets',
        'count': args.count, 'query_reformulation': False,
        'success': 'all frozen questions dispatch table; complete nonempty evidence matches direct call',
        'comparison': 'all native fields except timestamps/timings; float rtol=1e-5, atol=1e-6',
        'scope': 'integration parity only; no retrieval quality or answer quality claim',
    }
    write_json(output / 'protocol.json', protocol)
    summary = {'status': 'running', 'passed': 0, 'count': args.count}
    write_json(output / 'summary.json', summary)
    try:
        started = perf_counter()
        orchestrator = Orchestrator.from_config(config)
        summary['load_seconds'] = perf_counter() - started
        table = orchestrator.retrievers['table']
        with (output / 'traces.jsonl').open('x', encoding='utf-8') as stream:
            for record in selected:
                trace = {'router_id': record['router_id'], 'query': record['query']}
                try:
                    integrated = orchestrator.search(record['query'], query_id=record['router_id'])
                    trace['integrated'] = integrated
                    # Run table search directly to compare it with the connected system.
                    direct = table.pipeline.search(record['query'], table.table_top_k,
                                                   cell_top_k=table.cell_top_k,
                                                   query_id=record['router_id'])
                    trace['direct'] = direct
                    routed = integrated['modalities'].get('table')
                    trace['table_dispatched'] = routed is not None
                    trace['native_evidence_matches'] = bool(
                        routed and routed.get('native') and
                        equivalent(stable_evidence(routed['native']), stable_evidence(direct)))
                    trace['public_evidence_matches'] = bool(
                        routed and equivalent(routed['evidence'], direct['cells']))
                    trace['passed'] = bool(trace['native_evidence_matches'] and
                                           trace['public_evidence_matches'] and
                                           routed['status'] == direct['status'] == 'complete' and
                                           direct['cells'])
                except Exception as error:
                    trace.update(passed=False, error=str(error), error_type=type(error).__name__)
                summary['passed'] += trace['passed']
                stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(f"{record['router_id']}: {'PASS' if trace['passed'] else 'FAIL'}", flush=True)
        if file_hash(args.validation_file) != data_hash:
            raise ValueError('Validation input changed during smoke check')
        if manifest_hash and file_hash(args.validation_file.parent / 'manifest.json') != manifest_hash:
            raise ValueError('Validation manifest changed during smoke check')
        summary['status'] = 'passed' if summary['passed'] == args.count else 'failed'
    except Exception as error:
        summary.update(status='error', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        write_json(output / 'summary.json', summary)
        print(f"Smoke check: {summary['status']} ({summary['passed']}/{args.count}); {output}", flush=True)
    return output


if __name__ == '__main__':
    result = main()
    if json.loads((result / 'summary.json').read_text())['status'] != 'passed':
        raise SystemExit(1)
