# Loads runtime configuration and runs the full Stage 1 → Stage 2 retrieval pipeline.
import argparse
import json
from pathlib import Path

from .data import load_json
from .pipeline import EvidencePipeline
from .live.stage2.bge.model import DEFAULT_MODEL

# Local runtime settings; the example preserves the same defaults when none exists.
DEFAULT_CONFIG = Path(__file__).with_name('config.json')
# Required settings that must be present in config.json.
_FIELDS = {
    'stage1_index', 'stage2_checkpoint', 'stage2_max_length', 'device',
    'table_top_k', 'cell_top_k', 'table_source_dir',
}

# Loads and validates runtime settings and resolves configured file paths.
def load_config(path=DEFAULT_CONFIG):
    if Path(path) == DEFAULT_CONFIG and not DEFAULT_CONFIG.exists():
        path = DEFAULT_CONFIG.with_name('config.example.json')
    path = Path(path).expanduser().resolve()
    values = load_json(path)
    if not isinstance(values, dict) or set(values) != _FIELDS:
        raise ValueError(f'Config must contain exactly: {", ".join(sorted(_FIELDS))}')
    for key in ('table_top_k', 'cell_top_k', 'stage2_max_length'):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if not isinstance(values['device'], str) or not values['device'].strip():
        raise ValueError('device must be a nonempty string.')
    for key in ('stage1_index', 'stage2_checkpoint', 'table_source_dir'):
        value = values[key]
        if key == 'stage2_checkpoint' and value == DEFAULT_MODEL:
            continue
        if value is not None:
            values[key] = str((path.parent / value).resolve()) if not Path(value).is_absolute() else value
    return values

# Builds the full Stage 1 and Stage 2 retrieval pipeline from config.
def build_pipeline(config):
    """Initialize both models once from the configured, completed artifacts."""
    return EvidencePipeline.from_artifacts(
        stage1_index=config['stage1_index'],
        checkpoint=config['stage2_checkpoint'],
        stage2_max_length=config['stage2_max_length'],
        device=config['device'],
        table_source_dir=config['table_source_dir'],
    )

# Runs one query through the configured retrieval pipeline and prints the result.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', help='Natural-language question to search for')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='Stable retrieval config (default: table_retrieval/config.json)')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = build_pipeline(config).search(
            args.query, config['table_top_k'], cell_top_k=config['cell_top_k'])
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f'Retrieval failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
