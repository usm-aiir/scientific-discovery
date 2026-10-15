# Resolves evaluation settings into dataset paths, output paths, and runtime options.
from pathlib import Path
from types import SimpleNamespace

from ..data import digest, load_json
from ..stage1.artifacts import resolve_device


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent

# Builds the evaluation paths and runtime arguments for validation or test.
def arguments(settings, test=False):
    data = Path(settings['data'])
    split = 'test' if test else 'val'
    experiment = settings.get('experiment')
    legacy = Path(settings['test_output'] if test else settings['validation_output'])
    if experiment is None and legacy.exists():
        output = rankings = results = legacy
        source = Path(settings['validation_output']) if test else None
        embeddings = output / 'embeddings.npy'
    else:
        root = Path(settings.get('output_root', 'outputs'))
        name = experiment or 'default'
        output = root / 'indexes' / name / split
        rankings = root / 'rankings' / split / name
        results = root / 'results' / name / split
        source = root / 'indexes' / name / 'val' if test else None
        embeddings = root / 'embeddings' / name / 'corpus.npy'
    return SimpleNamespace(
        corpus=data / 'Corpus.json', queries=data / ('Test.json' if test else 'Val.json'),
        qrels=Path(settings.get(f'{split}_qrels', data / (
            'Test_table_qrels.INSTRUCTOR_ONLY.tsv' if test else 'Val_table_qrels.tsv'))),
        output=output, rankings=rankings, results=results, source=source,
        embeddings=embeddings, model=settings['model'], revision=settings['revision'],
        max_length=settings['max_length'], depth=settings['candidate_depth'],
        top_k=settings.get('top_k', 10), rrf_constant=settings.get('fusion', {}).get('constant', 60),
        batch_size=settings['batch_size'], device=resolve_device(settings['device']))

# Validates evaluation settings and checks that saved artifacts still match them.
def check_settings(settings, args):
    if not 4 <= args.max_length <= 8192 or args.batch_size < 1 or args.depth < max(10, args.top_k) or args.top_k < 10:
        raise ValueError('Require max_length 4..8192, positive batch_size, and candidate_depth >= top_k >= 10')
    if args.rrf_constant < 1:
        raise ValueError('Fusion constant must be positive')
    manifest = args.output / 'prepared.json'
    if not manifest.exists():
        return
    saved = load_json(manifest)
    expected = dict(model=args.model, revision=args.revision, max_length=args.max_length,
                    candidate_depth=args.depth, top_k=args.top_k, rrf_constant=args.rrf_constant)
    for key, value in expected.items():
        if saved.get(key, {'top_k': 10}.get(key)) != value:
            raise ValueError(f'{key} differs from the saved evaluation; choose a new --experiment name')
    for path, key in [(args.corpus, 'corpus_sha256'), (args.queries, 'queries_sha256'),
                      (args.qrels, 'qrels_sha256'),
                      (args.output / 'table_inputs.json', 'inputs_sha256'),
                      (args.output / 'table_ids.json', 'ids_sha256'),
                      (args.output / 'queries.json', 'saved_queries_sha256')]:
        if digest(path) != saved[key]:
            raise ValueError(f'{path} differs from the saved evaluation; choose a new --experiment name')
