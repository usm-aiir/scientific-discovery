# Runs the saved table system on a JSON list of questions.
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from table_retrieval.data import RESULTS_DIR, digest, load_json, write_json, write_predictions
from table_retrieval.experiments.stage1.evaluate import load_queries
from table_retrieval.search import build_pipeline, load_config

SETTINGS = Path(__file__).with_name('settings.json')


# Searches every question and keeps failures alongside successful results.
def predictions(engine, queries, config):
    for position, item in enumerate(queries, 1):
        query_id = str(item['query_id'])
        try:
            result = engine.search(item['query'], config['table_top_k'],
                                   cell_top_k=config['cell_top_k'], query_id=query_id)
        except Exception as error:
            result = {'query_id': query_id, 'query': item['query'], 'status': 'error',
                      'tables': [], 'cells': [],
                      'errors': [{'error': str(error), 'error_type': type(error).__name__}]}
        print(f"{position}/{len(queries)} {query_id}: {result['status']}", flush=True)
        yield result


# Loads the models once and saves a new batch without reading relevance labels.
def run(input_path, *, config_path=SETTINGS, output=None, device=None):
    input_path, config_path = Path(input_path).resolve(), Path(config_path).resolve()
    hashes = {'input': digest(input_path), 'settings': digest(config_path)}
    queries = load_queries(input_path)
    config = load_config(config_path)
    if device is not None:
        config['device'] = device
    if hashes != {'input': digest(input_path), 'settings': digest(config_path)}:
        raise ValueError('Questions or settings changed while loading')
    output = Path(output) if output is not None else RESULTS_DIR / (
        'table_retrieval_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'questions': len(queries)}
    started = perf_counter()
    try:
        write_json(output / 'protocol.json', {
            'input': str(input_path), 'settings': str(config_path), 'sha256': hashes,
            'configuration': config, 'questions': len(queries),
            'scope': 'retrieval only; no router, rewriting, judge, training or relevance labels',
        })
        checkpoint = Path(config['stage2_checkpoint'])
        if not checkpoint.is_dir():
            raise FileNotFoundError('Download the saved models first: python -m table_retrieval.assets download')
        write_json(output / 'checkpoint.json', {
            'directory': str(checkpoint), 'sha256': {
                name: digest(checkpoint / name) for name in (
                    'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json')},
        })
        loading = perf_counter()
        engine = build_pipeline(config)
        summary['load_seconds'] = perf_counter() - loading
        write_json(output / 'retrieval_settings.json', engine.configuration(
            config['table_top_k'], config['cell_top_k']))
        count, failed = write_predictions(output / 'predictions.jsonl',
                                           predictions(engine, queries, config))
        if hashes != {'input': digest(input_path), 'settings': digest(config_path)}:
            raise ValueError('Questions or settings changed during retrieval')
        summary.update(status='completed_with_errors' if failed else 'completed',
                       written=count, failed=failed)
    except Exception as error:
        summary.update(status='error', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        summary['total_seconds'] = perf_counter() - started
        write_json(output / 'summary.json', summary)
        print(f"{summary['status']}: {output}", flush=True)
    return output


# Runs the downloaded table models on the supplied question file.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='JSON list with query_id and query')
    parser.add_argument('--config', default=SETTINGS, type=Path)
    parser.add_argument('--output', type=Path, help='New results folder')
    parser.add_argument('--device', help='Override the device, for example cuda:0 or cpu')
    args = parser.parse_args(argv)
    try:
        output = run(args.input, config_path=args.config, output=args.output, device=args.device)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'{error}\n')
    if load_json(output / 'summary.json')['status'] != 'completed':
        parser.exit(1, 'Some questions failed; details were saved with the results.\n')
    return output


if __name__ == '__main__':
    main()
