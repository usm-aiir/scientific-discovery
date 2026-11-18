"""Read routing data and describe the table, figure and text route choices."""

import hashlib
import json
from pathlib import Path

ROUTING_TASK = 'reviewed_routing'
MODALITIES = ['table', 'figure', 'text']
ROUTES = [['table'], ['figure'], ['text'], ['table', 'figure'],
          ['table', 'text'], ['figure', 'text'], ['table', 'figure', 'text']]
ROUTE_LABELS = ['+'.join(route) for route in ROUTES]
ROUTE_IDS = {frozenset(route): i for i, route in enumerate(ROUTES)}
CLEAN_FIELDS = {'router_id', 'query', 'acceptable_sets', 'source_split'}


def file_hash(path):
    """Make a fingerprint we can use to check whether a file has changed."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Save the given information as a JSON file."""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )


def load_records(path, split):
    """Read a training or evaluation file and check that its examples look right."""
    if split not in ('Train', 'Val', 'Test'):
        raise ValueError('Expected Train, Val or Test')
    path = Path(path)
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(str(path) + ': expected a nonempty JSON array')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != CLEAN_FIELDS:
            raise ValueError(str(path) + ': expected exactly the four clean fields; metadata is not model input')
        if row['source_split'] != split:
            raise ValueError(str(path) + ': expected only ' + split + ' records')
        if not isinstance(row['query'], str) or not row['query'].strip():
            raise ValueError('Every record needs a nonempty query')
        rid = row['router_id']
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError('router_id must be a unique nonempty string')
        seen.add(rid)
        route_ids(row['acceptable_sets'])
    return rows


def configure_classifier(config, max_length):
    """Give the model our route names, route order and question-length limit."""
    config.num_labels = len(ROUTES)
    config.id2label = dict(enumerate(ROUTE_LABELS))
    config.label2id = {label: i for i, label in enumerate(ROUTE_LABELS)}
    config.problem_type = 'single_label_classification'
    config.router_task = ROUTING_TASK
    config.router_max_length = max_length
    config.router_target_semantics = 'Choose one known acceptable route; outer OR, inner AND'
    config.router_route_order = [route.copy() for route in ROUTES]


def checkpoint_task(config):
    """Check that this model uses the routing setup we trained it with."""
    try:
        labels = {int(key): value for key, value in config.id2label.items()}
    except (AttributeError, TypeError, ValueError):
        raise ValueError('Checkpoint lacks a valid seven-action router label map')
    if (getattr(config, 'router_task', None) != ROUTING_TASK
            or labels != dict(enumerate(ROUTE_LABELS))
            or getattr(config, 'num_labels', len(ROUTES)) != len(ROUTES)
            or getattr(config, 'router_route_order', None) != ROUTES
            or type(getattr(config, 'router_max_length', None)) is not int
            or config.router_max_length <= 0):
        raise ValueError('Expected a reviewed seven-action routing checkpoint with the saved route order and query length')
    return ROUTING_TASK


def evaluation_coverage(path, split):
    """Check the file matches the review record, then report how much was reviewed."""
    path = Path(path)
    manifest_path = path.parent / 'manifest.json'
    if not manifest_path.is_file():
        return None, None
    digest = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    info = manifest.get('splits', {}).get(split)
    if not info or info.get('file') != path.name:
        return None, None
    if manifest.get('output_sha256', {}).get(path.name) != file_hash(path):
        raise ValueError('Evaluation file differs from its manifest')
    if file_hash(manifest_path) != digest:
        raise ValueError('Evaluation manifest changed while loading')
    return {key: info[key] for key in ('source_count', 'reviewed', 'accepted', 'pending', 'excluded', 'review_complete') if key in info}, digest


def check_test_coverage(path, allow_partial=False):
    """Check whether we are allowed to evaluate Test data based on its review status."""
    info, _ = evaluation_coverage(path, 'Test')
    if not allow_partial and (not info or not info.get('review_complete') or info.get('pending', 0)):
        raise ValueError('Test review is incomplete or unverified. Keep it held out; --allow-partial-test explicitly permits scoring the accepted partial snapshot.')
    return info


def describe_records(rows):
    """Count the examples and the route choices they contain."""
    counts = dict.fromkeys(ROUTE_LABELS, 0)
    for row in rows:
        for i in route_ids(row['acceptable_sets']):
            counts[ROUTE_LABELS[i]] += 1
    return {'count': len(rows), 'accepted_route_counts': counts,
            'questions_with_alternatives': sum(len(r['acceptable_sets']) > 1 for r in rows),
            'joint_only_questions': sum(all(len(s) > 1 for s in r['acceptable_sets']) for r in rows),
            'model_input_fields': ['query']}


def route_ids(acceptable_sets):
    """Turn the accepted route choices into the model's route numbers."""
    if not isinstance(acceptable_sets, list) or not acceptable_sets:
        raise ValueError('acceptable_sets must contain at least one sufficient route')
    ids = []
    for route in acceptable_sets:
        if (not isinstance(route, list) or not route
                or any(not isinstance(m, str) or m not in MODALITIES for m in route)
                or len(set(route)) != len(route)):
            raise ValueError('Each route must contain unique table, figure, or text modalities')
        index = ROUTE_IDS[frozenset(route)]
        if index in ids:
            raise ValueError('Duplicate acceptable route')
        ids.append(index)
    return ids


def target_mask(record):
    """Mark which routing choices are accepted for this question."""
    accepted = set(route_ids(record['acceptable_sets']))
    return [index in accepted for index in range(len(ROUTES))]


def collate_reviewed(batch):
    # Keep each question's accepted route choices together when making a batch.
    """Put examples into a batch while keeping all their accepted route choices."""
    import torch
    return {'query': [r['query'] for r in batch],
            'acceptable_mask': torch.tensor([target_mask(r) for r in batch], dtype=torch.bool)}


def routing_metrics(records, predictions):
    """Score the router's choices against the accepted routes."""
    if not records or len(records) != len(predictions):
        raise ValueError('Metrics need equal nonempty target and prediction sequences')
    exact, covered, calls, excess = 0, 0, 0, 0
    counts = dict.fromkeys(ROUTE_LABELS, 0)
    groups = {key: {'count': 0, 'known_acceptable_route_matches': 0, 'sufficient_set_covered': 0}
              for key in ('single_option', 'alternative_options', 'joint_only')}
    for row, index in zip(records, predictions):
        if type(index) is not int or not 0 <= index < len(ROUTES):
            raise ValueError('Invalid predicted route ID')
        ids = route_ids(row['acceptable_sets'])
        acceptable_routes = [set(ROUTES[i]) for i in ids]
        predicted = set(ROUTES[index])
        is_exact = index in ids
        sufficient = [route for route in acceptable_routes if route <= predicted]
        is_covered = bool(sufficient)
        exact += is_exact
        covered += is_covered
        calls += len(predicted)
        excess += min((len(predicted - route) for route in sufficient), default=0)
        counts[ROUTE_LABELS[index]] += 1
        if all(len(route) > 1 for route in acceptable_routes):
            kind = 'joint_only'
        elif len(ids) > 1:
            kind = 'alternative_options'
        else:
            kind = 'single_option'
        group = groups[kind]
        group['count'] += 1
        group['known_acceptable_route_matches'] += is_exact
        group['sufficient_set_covered'] += is_covered
    n = len(records)
    return dict(count=n, known_acceptable_route_match_rate=exact / n,
                sufficient_set_coverage=covered / n, mean_selected_modalities=calls / n,
                mean_extra_modalities_when_covered=excess / covered if covered else None,
                predicted_route_counts=counts, by_target_kind=groups,
                scope='Agreement with known sufficient route sets; unlisted routes are unverified, not established negatives.')
