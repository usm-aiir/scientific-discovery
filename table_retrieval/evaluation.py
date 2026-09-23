"""Validate saved rankings, calculate Recall@10 and compare complementary hits."""
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from .data import load_json, load_queries, normalize_table_id, check_id, digest, read_run, write_json, load_qrels

def evaluate(run_path, qrels_path, query_ids, k=10, min_grade=1):
    """Score all requested queries, including queries absent from the run.

    Grades >= min_grade count as relevant. Queries without positive judgments
    are reported separately and excluded from the mean. Missing run entries
    for judged queries receive zero recall. Corpus and qrels table ID formats
    are normalized in memory before matching and duplicate detection.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    query_ids = [str(qid) for qid in query_ids]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate query IDs.")
    relevant = load_qrels(qrels_path, min_grade)
    runs = defaultdict(list)
    seen = set()
    allowed = set(query_ids)
    for line in Path(run_path).read_text().splitlines():
        if not line.strip():
            continue
        qid, placeholder, uid, rank, score, _ = line.split()
        uid = normalize_table_id(uid)
        rank, score = int(rank), float(score)
        if qid not in allowed or placeholder != "Q0" or rank < 1 or not math.isfinite(score):
            raise ValueError(f"Invalid TREC line: {line}")
        if (qid, uid) in seen:
            raise ValueError(f"Duplicate query/table pair: {qid}, {uid}")
        seen.add((qid, uid))
        runs[qid].append((rank, score, uid))
    per_query = {}
    unjudged = []
    for qid in query_ids:
        ranked = sorted(runs[qid])
        if [r for r, _, _ in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError(f"Non-consecutive ranks for query {qid}")
        if any(a[1] < b[1] for a, b in zip(ranked, ranked[1:])):
            raise ValueError(f"Scores increase with rank for query {qid}")
        if not relevant[qid]:
            unjudged.append(qid)
            continue
        retrieved = {uid for _, _, uid in ranked[:k]}
        per_query[qid] = len(retrieved & relevant[qid]) / len(relevant[qid])
    return {f"recall@{k}": sum(per_query.values()) / len(per_query) if per_query else None,
            "evaluated_queries": len(per_query), "min_relevance_grade": min_grade,
            "queries_without_positive_judgments": unjudged, "per_query": per_query}


def assess(args):
    rankings = getattr(args, 'rankings', args.output)
    results = getattr(args, 'results', args.output)
    manifest = load_json(args.output / 'prepared.json')
    if digest(manifest['qrels']) != manifest['qrels_sha256']:
        raise ValueError('Judgments changed since preparation')
    queries = load_queries(args.output / 'queries.json')
    if digest(args.output / 'queries.json') != manifest['saved_queries_sha256']:
        raise ValueError('Prepared queries changed')
    qids = [check_id(q['query_id']) for q in queries]
    ranks = read_run(rankings / 'bge.run') if (rankings / 'bge.run').exists() else {}
    lexical = read_run(rankings / 'bm25_matched.run') if (rankings / 'bm25_matched.run').exists() else {}
    clipped = args.output / 'truncated_queries.json'
    query_clipped = load_json(clipped) if clipped.exists() else []
    reports = {}
    for name in ['bm25_full', 'bm25_matched', 'bge', 'fusion']:
        if not (rankings / f'{name}.run').exists():
            continue
        report = evaluate(rankings / f'{name}.run', manifest['qrels'], qids, k=10)
        write_json(results / f'{name}.metrics.json', report)
        reports[name] = report
    if len(reports) < 4:
        print(json.dumps({name: report['recall@10'] for name, report in reports.items()}, indent=2))
        return
    relevant = load_qrels(manifest['qrels'])
    lexical_only, dense_only, union_recalls = [], [], []
    for qid in ranks:
        gold = relevant[qid]
        if not gold:
            continue
        a = {normalize_table_id(uid) for uid, _ in lexical.get(qid, [])[:10]} & gold
        b = {normalize_table_id(uid) for uid, _ in ranks[qid][:10]} & gold
        if a - b: lexical_only.append(qid)
        if b - a: dense_only.append(qid)
        union_recalls.append(len(a | b) / len(gold))
    summary = {'recall@10': {name: report['recall@10'] for name, report in reports.items()},
               'queries_with_positives_unique_to_bm25_top10': lexical_only,
               'queries_with_positives_unique_to_bge_top10': dense_only,
               'union_top10_recall_up_to_20_results_not_a_top10_score': float(np.mean(union_recalls)),
               'truncated_tables': manifest['truncated_tables'], 'truncated_queries': query_clipped,
               'rrf': {'constant': manifest['rrf_constant'], 'candidate_depth_per_method': manifest['candidate_depth'],
                       'weights': [1, 1], 'output_k': manifest.get('top_k', 10), 'tuned_on_validation': False}}
    write_json(results / 'comparison.json', summary)
    print(json.dumps({'recall@10': summary['recall@10'],
                      'queries_with_unique_bm25_positives': len(lexical_only),
                      'queries_with_unique_bge_positives': len(dense_only),
                      'truncated_tables': manifest['truncated_tables'],
                      'truncated_queries': len(query_clipped),
                      'details': str(results / 'comparison.json')}, indent=2))


def inspect_misses(output):
    manifest = load_json(output / 'prepared.json')
    queries = load_json(output / 'queries.json')
    metrics = load_json(output / 'fusion.metrics.json')
    methods = {name: read_run(output / f'{name}.run')
               for name in ['bm25_matched', 'bge', 'fusion']}
    relevant = load_qrels(manifest['qrels'])
    corpus_ids = {normalize_table_id(uid) for uid in load_json(output / 'table_ids.json')}
    misses = []
    for query in queries:
        qid = check_id(query['query_id'])
        if metrics['per_query'].get(qid, 1) >= 1:
            continue
        ranks = {name: {normalize_table_id(uid): i for i, (uid, _) in enumerate(run.get(qid, []), 1)}
                 for name, run in methods.items()}
        missing = relevant[qid] - set(ranks['fusion'])
        misses.append({'query_id': qid, 'query': query['query'], 'fusion_recall@10': metrics['per_query'][qid],
                       'missing_positives': [{'table_id': uid, 'present_in_corpus': uid in corpus_ids,
                           'bm25_rank_in_top100': ranks['bm25_matched'].get(uid),
                           'bge_rank_in_top100': ranks['bge'].get(uid)} for uid in sorted(missing)],
                       'fusion_top10': methods['fusion'].get(qid, [])})
    write_json(output / 'misses.json', {'queries_with_misses': len(misses), 'queries': misses,
                         'note': 'Unjudged retrieved tables are not confirmed irrelevant. No test-driven tuning.'})
