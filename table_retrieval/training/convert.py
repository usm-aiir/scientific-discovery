"""Strictly convert the official TensorFlow DTR checkpoint to PyTorch.

Requires tensorflow-cpu only during conversion. Every inference tensor must map;
optimizer slots are excluded. Source-file SHA-256 hashes are recorded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import BertTokenizer, TapasConfig, TapasModel, TapasTokenizer

from ..retrievers.tapas import PaperRetriever


def tf_name(name, scope):
    if name.startswith('embeddings.'):
        name = name.replace('.weight', '') if 'LayerNorm' not in name else name
    name = re.sub(r'encoder\.layer\.(\d+)', r'encoder.layer_\1', name)
    if name.endswith('LayerNorm.weight'):
        name = name[:-6] + 'gamma'
    elif name.endswith('LayerNorm.bias'):
        name = name[:-4] + 'beta'
    elif name.endswith('.weight'):
        name = name[:-6] + 'kernel'
    return scope + '/' + name.replace('.', '/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/dtr_paper/pretrained'))
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Output must be new or empty')
    import tensorflow as tf
    config_path = args.source / 'bert_config.json'
    config_data = json.loads(config_path.read_text())
    type_sizes = config_data.pop('type_vocab_size')
    if not isinstance(type_sizes, list) or len(type_sizes) != 7:
        raise ValueError('Expected seven structural embedding tables')
    # Google's BERT uses the tanh approximation; HF's "gelu" uses erf.
    if config_data.get('hidden_act') == 'gelu':
        config_data['hidden_act'] = 'gelu_new'
    config = TapasConfig(**config_data, type_vocab_sizes=type_sizes,
                         reset_position_index_per_cell=False, layer_norm_eps=1e-12)
    prefix = args.source / 'model.ckpt'
    reader = tf.train.load_checkpoint(str(prefix))
    available = reader.get_variable_to_shape_map()
    mapped = set()
    encoders = {}
    # The official model creates the table tower first (bert), then query (bert_1).
    for side, scope in [('table', 'bert'), ('query', 'bert_1')]:
        encoder = TapasModel(config)
        for name, parameter in encoder.named_parameters():
            source_name = tf_name(name, scope)
            if source_name not in available:
                raise ValueError(f'Missing tensor {source_name} for {side}.{name}')
            array = reader.get_tensor(source_name)
            if source_name.endswith('/kernel'):
                array = array.T
            if tuple(array.shape) != tuple(parameter.shape):
                raise ValueError(f'Shape mismatch: {source_name}: {array.shape} != {parameter.shape}')
            parameter.data.copy_(torch.from_numpy(np.asarray(array).copy()))
            mapped.add(source_name)
        encoders[side] = encoder
        print(f'Converted {side} encoder', flush=True)
    vocab = str(args.source / 'vocab.txt')
    model = PaperRetriever(encoders['query'], encoders['table'], BertTokenizer(vocab_file=vocab),
                           TapasTokenizer(vocab_file=vocab, cell_trim_length=-1,
                                          max_row_id=type_sizes[2], max_column_id=type_sizes[1]))
    for side, source_name in [('query', 'text_projection'), ('table', 'table_projection')]:
        array = reader.get_tensor(source_name)
        parameter = getattr(model, f'{side}_projection').weight
        if tuple(array.shape) != tuple(parameter.shape):
            raise ValueError(f'Invalid projection shape: {source_name}')
        parameter.data.copy_(torch.from_numpy(array.copy()))
        mapped.add(source_name)
    unused = sorted(name for name in available if name not in mapped and name != 'global_step'
                    and not any(part in name for part in ('adam_m', 'adam_v', 'AdamWeightDecayOptimizer')))
    if unused:
        raise ValueError(f'Unmapped non-optimizer variables: {unused}')
    model.save(args.output, {'source': str(args.source), 'retrieval_pretrained': True,
                            'source_url': 'https://storage.googleapis.com/tapas_models/2021_04_27/tapas_dual_encoder_proj_256_large.zip',
                            'converted_inference_tensors': len(mapped)})
    checksums = {}
    for path in [config_path, args.source / 'vocab.txt', args.source / 'model.ckpt.index',
                 *sorted(args.source.glob('model.ckpt.data-*'))]:
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        checksums[path.name] = digest.hexdigest()
    (args.output / 'conversion.json').write_text(json.dumps({'mapped_tensors': sorted(mapped),
        'tensorflow_version': tf.__version__, 'source_sha256': checksums,
        'config_sha256': checksums[config_path.name]}, indent=2) + '\n')
    print(f'Saved {args.output}; {len(mapped)} inference tensors mapped, no unexplained tensors.')


if __name__ == '__main__':
    main()
