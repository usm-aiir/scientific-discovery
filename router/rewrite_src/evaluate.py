"""Check whether rewriting questions helps us find better tables and cells."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from router_src.data import file_hash, write_json
from .rewrite import GemmaRewriter
from pipeline.search import Orchestrator, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('settings.json')


def latency_summary(seconds):
    """Summarize how long the searches took."""
    if not seconds:
        return None
    median, p95 = np.percentile(seconds, [50, 95])
    return {'median': float(median), 'p95': float(p95)}


def score_result(native, gold, relevant_tables, k=5):
    """Compare the tables and cells we found with the ones marked as relevant."""
    from table_retrieval.experiments.stage2_bge.metrics import ndcg_at_k

    ranking = [(c['table_id'], c['cell_id']) for c in native['cells']]
    if len(ranking) != len(set(ranking)):
        raise ValueError('Duplicate source cell in ranking')
    tables = [t['table_id'] for t in native['tables']]
    correct = len(set(ranking[:k]) & gold)
    return {
        'correct_cells': correct, 'returned_cells': len(ranking[:k]), 'gold_cells': len(gold),
        'cell_precision@5': correct / len(ranking[:k]) if ranking else 0.0,
        'cell_recall@5': correct / len(gold),
        'cell_ndcg@5': ndcg_at_k(ranking, gold, k),
        'cell_mrr': next((1 / i for i, key in enumerate(ranking, 1) if key in gold), 0.0),
        'gold_cell_coverage': len(set(ranking) & gold) / len(gold),
        'table_recall@5': len(set(tables[:k]) & relevant_tables) / len(relevant_tables),
        'table_ndcg@5': ndcg_at_k(tables, relevant_tables, k),
        'table_mrr@5': next((1 / i for i, uid in enumerate(tables[:k], 1)
                              if uid in relevant_tables), 0.0),
    }


def summarize(traces):
    """Put the scores together so we can compare original and rewritten questions."""
    metrics = {}
    for arm in ('original', 'rewritten'):
        scores = [t[arm]['scores'] for t in traces]
        hits = sum(s['correct_cells'] for s in scores)
        returned = sum(s['returned_cells'] for s in scores)
        gold = sum(s['gold_cells'] for s in scores)
        metrics[arm] = {
            'queries': len(traces), 'correct_cells': hits, 'returned_cells': returned,
            'gold_cells': gold, 'cell_precision@5': hits / returned if returned else 0.0,
            'cell_recall@5': hits / gold,
        }
        for key in ('cell_ndcg@5', 'cell_mrr', 'gold_cell_coverage',
                    'table_recall@5', 'table_ndcg@5', 'table_mrr@5'):
            metrics[arm][key] = float(np.mean([s[key] for s in scores]))
        metrics[arm]['latency_seconds'] = latency_summary([t[arm]['seconds'] for t in traces])
    paired = {}
    rng = np.random.default_rng(42)
    for key in ('cell_ndcg@5', 'cell_recall@5', 'table_recall@5'):
        deltas = np.array([t['rewritten']['scores'][key] - t['original']['scores'][key]
                           for t in traces])
        # Weight questions by their marked cell counts, just as the overall recall does.
        weights = np.array([t['original']['scores']['gold_cells'] for t in traces])
        indices = rng.integers(0, len(traces), size=(2000, len(traces)))
        if key == 'cell_recall@5':
            samples = np.average(deltas[indices], weights=weights[indices], axis=1)
        else:
            samples = deltas[indices].mean(axis=1)
        paired[key] = {
            'aggregate_delta': metrics['rewritten'][key] - metrics['original'][key],
            'wins': int((deltas > 1e-12).sum()), 'losses': int((deltas < -1e-12).sum()),
            'ties': int((abs(deltas) <= 1e-12).sum()),
            'paired_bootstrap_95_ci': list(map(float, np.percentile(samples, [2.5, 97.5]))),
        }
    fallback = sum(t.get('reformulation', {}).get('status') == 'fallback' for t in traces)
    rewrite_errors = sum(t.get('reformulation', {}).get('fallback_kind') == 'error' for t in traces)
    failures = sum(t[arm]['native']['status'] not in ('complete', 'not_dispatched')
                   for t in traces for arm in ('original', 'rewritten'))
    dispatched = sum('table' in t['routing']['route'] for t in traces)
    success = (metrics['rewritten']['cell_ndcg@5'] > metrics['original']['cell_ndcg@5']
               and metrics['rewritten']['cell_recall@5'] >= metrics['original']['cell_recall@5']
               and metrics['rewritten']['table_recall@5'] >= metrics['original']['table_recall@5']
               and rewrite_errors == 0 and failures == 0)
    return {'status': 'completed' if not (failures or rewrite_errors) else 'completed_with_errors',
            'metrics': metrics, 'paired': paired, 'table_dispatched': dispatched,
            'rewrite_fallbacks': fallback, 'retrieval_errors': failures,
            'guard_rejections': fallback - rewrite_errors, 'rewrite_errors': rewrite_errors,
            'identity_rewrites': sum(t.get('reformulation', {}).get('status') == 'identity'
                                     for t in traces),
            'rewrite_latency_seconds': latency_summary([
                t['reformulation']['seconds'] for t in traces if 'reformulation' in t
            ]),
            'meets_predeclared_success_rule': success,
            'interpretation': 'validation evidence only; inspect confidence intervals and error traces'}


def compare_question(engine, query, qid, *, rewrite_first=False):
    """Try both search queries while keeping the same routing choice."""
    started = perf_counter()
    decision = engine.router.route(query)
    routing_seconds = perf_counter() - started
    trace = {'query_id': qid, 'query': query, 'routing': decision,
             'routing_seconds': routing_seconds,
             'arm_order': ['rewritten', 'original'] if rewrite_first else ['original', 'rewritten']}
    if 'table' not in decision['route']:
        for arm in trace['arm_order']:
            trace[arm] = {'native': {'cells': [], 'tables': [], 'errors': [],
                                     'status': 'not_dispatched'}, 'seconds': routing_seconds}
        return trace
    table = engine.retrievers['table']
    for arm in trace['arm_order']:
        before = perf_counter()
        try:
            stage1_query = query
            if arm == 'rewritten':
                rewrite = engine.rewriter.rewrite(query, decision['route'])
                trace['reformulation'] = rewrite
                stage1_query = rewrite['query']
            # Keep the full list so we can score correct cells after the first five.
            native = table.pipeline.search(query, table.table_top_k, cell_top_k=2**31 - 1,
                                           query_id=qid, stage1_query=stage1_query)
        except Exception as error:
            native = {'cells': [], 'tables': [], 'status': 'error',
                      'errors': [{'error': str(error), 'error_type': type(error).__name__}]}
        trace[arm] = {'native': native, 'seconds': perf_counter() - before + routing_seconds}
    return trace


def main(argv=None):
    """Run both searches on validation questions and save their scores and results."""
    from table_retrieval.data import DEFAULT_DATA_DIR, table_split_paths, digest
    from table_retrieval.experiments.stage1.metrics import load_qrels
    from table_retrieval.experiments.stage1.evaluate import load_queries
    from table_retrieval.experiments.stage2_bge.dataset import load_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--limit', type=int, help='First N Val questions, for a smoke run only')
    parser.add_argument('--output', type=Path, help='New output directory')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    config = load_config(args.config)
    if 'rewriter' not in config:
        parser.error('Use a configuration with a rewriter')
    if (config['table_top_k'], config['cell_top_k'], config['stage2_max_length']) != (5, 5, 512):
        parser.error('This benchmark requires table_top_k=5, cell_top_k=5 and stage2_max_length=512')
    if config['table_source_dir'] is not None:
        parser.error('Use the frozen structural_v1 benchmark without optional source HTML')
    output = args.output or (ROOT / 'results') / (
        'gemma_rewriting_val_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        parser.error('Output exists; choose a new directory')
    paths = table_split_paths(args.data_dir, 'val')  # No Test path is opened.
    queries = load_queries(paths['queries'])
    selected = queries if args.limit is None else queries[:args.limit]
    sources = {**paths, 'config': args.config,
               'index_manifest': Path(config['stage1_index']) / 'prepared.json',
               'index_metadata': Path(config['stage1_index']) / 'index.json',
               'stage2_config': Path(config['stage2_checkpoint']) / 'config.json'}
    hashes = {key: file_hash(path) for key, path in sources.items()}
    prepared = json.loads(sources['index_manifest'].read_text())
    corpus = Path(args.data_dir) / 'Corpus.json'
    if digest(corpus) != prepared['corpus_sha256']:
        parser.error('Validation corpus differs from the indexed corpus')
    hashes['corpus'] = prepared['corpus_sha256']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'questions.json', selected)
    write_json(output / 'protocol.json', {
        'split': 'val', 'configuration': config, 'input_sha256': hashes,
        'selection': 'all Val questions, or first N in unchanged source order before inference',
        'count': len(selected), 'smoke_only': args.limit is not None,
        'routing': 'original question routed once; same decision in both arms; missed table routes score zero',
        'arms': ['original Stage 1 query', 'Gemma Stage 1 query, original question for Stage 2'],
        'warmup': 'one first-question comparison excluded from measurements; alternate arm order thereafter',
        'cache_policy': 'same loaded models/index; no query-result cache; OS/device caches remain warm',
        'cell_metrics': 'micro P/R; macro binary nDCG@5; MRR over complete candidate-cell ranking',
        'table_metrics': 'macro binary recall/nDCG@5 and truncated MRR@5, not fusion Top-10 metrics',
        'judgments': 'original Val labels; all gold retained, unjudged retrieved items get no positive credit',
        'success': 'higher cell nDCG@5, no loss of cell recall@5 or table recall@5, no model/retrieval errors',
        'fallback': 'rejected constraints use the original query; report guard rejections separately from errors',
        'uncertainty': 'paired bootstrap 2000 resamples, seed 42; intervals descriptive, no test selection',
        'latency': 'routing included in both arms; rewrite included only in rewritten arm; loading excluded',
        'scope': 'table branch only; no answer-quality or figure/text retrieval claim',
    })
    summary = {'status': 'loading', 'count': len(selected)}
    write_json(output / 'summary.json', summary)
    try:
        GemmaRewriter(config['rewriter']).check_environment()
        engine = Orchestrator.from_config(config)
        engine.rewriter.load()
        all_queries, labels, _ = load_split(args.data_dir, 'val', engine.retrievers['table'].pipeline.retrieval.corpus)
        if all_queries != queries:
            raise ValueError('Validation questions changed before evaluation')
        relevant = load_qrels(paths['qrels'])
        # Match table IDs in the results to the IDs used in the labels.
        from table_retrieval.data import normalize_table_id
        warmup = compare_question(engine, selected[0]['query'], str(selected[0]['query_id']))
        write_json(output / 'warmup.json', warmup)
        traces = []
        with (output / 'traces.jsonl').open('x', encoding='utf-8') as stream:
            for index, row in enumerate(selected):
                qid = str(row['query_id'])
                trace = compare_question(engine, row['query'], qid, rewrite_first=bool(index % 2))
                gold_tables = {uid for (uid, _) in labels[qid]
                               if normalize_table_id(uid) in relevant[qid]}
                for arm in ('original', 'rewritten'):
                    trace[arm]['scores'] = score_result(trace[arm]['native'], labels[qid], gold_tables)
                traces.append(trace)
                stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                state = trace.get('reformulation', {}).get('status', 'table not dispatched')
                print(f'{index + 1}/{len(selected)}: {qid} ({state})', flush=True)
        for key, path in sources.items():
            if file_hash(path) != hashes[key]:
                raise ValueError(f'Input changed during evaluation: {path}')
        if digest(corpus) != hashes['corpus']:
            raise ValueError('Corpus changed during evaluation')
        summary = summarize(traces)
        summary['smoke_only'] = args.limit is not None
        write_json(output / 'metrics.json', summary)
    except Exception as error:
        summary.update(status='error', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        write_json(output / 'summary.json', summary)
        print(f"Rewriting comparison: {summary['status']}; {output}", flush=True)
    return output


if __name__ == '__main__':
    main()
