# Verifies metric definitions, frozen data, and train/save/reload compatibility.
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from table_retrieval.data import load_json
from table_retrieval.experiments.stage2_bge.data import (
    HERE, candidate_cells, query_labels, read_data, training_pairs,
)
from table_retrieval.experiments.stage2_bge.evaluate import validate_rankings
from table_retrieval.experiments.stage2_bge.metrics import score_queries, validation_ndcg
from table_retrieval.experiments.stage2_bge.prepare import validate_data


class ExperimentTests(unittest.TestCase):
    def test_fixed_data_and_reference_results(self):
        self.assertEqual(validate_data(), {'train': 1223, 'val': 122, 'test': 122, 'runtime_train': 975})
        expected = load_json(HERE / 'reference' / 'metrics.json')
        for split in ('val', 'test'):
            saved = load_json(HERE / 'reference' / f'{split}_rankings.json')['records']
            validate_rankings(saved, read_data(f'{split}_candidates.json'), query_labels(split))
            self.assertEqual(score_queries(saved), expected[split])
        self.assertAlmostEqual(expected['test']['recall@5'], 105 / 333)
        self.assertAlmostEqual(expected['test']['precision@5'], 105 / 455)

    def test_ndcg_uses_rank_order_and_includes_missing_gold(self):
        record = {'gold': [['t', 'a'], ['missing', 'b']], 'ranking': [['t', 'wrong'], ['t', 'a']],
                  'candidate_table_ids': ['t']}
        actual = score_queries([record], k=2)
        self.assertEqual(actual['recall@2'], .5)
        self.assertAlmostEqual(actual['ndcg@2'], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        record['ranking'].reverse()
        self.assertGreater(score_queries([record], k=2)['ndcg@2'], actual['ndcg@2'])
        self.assertEqual(validation_ndcg([(['a', 'b'], {'a', 'b'})]), 1)

    def test_negative_cells_are_never_training_gold(self):
        examples = read_data('train.json')['annotations']
        config = load_json(HERE / 'config.json')
        pairs, _ = training_pairs(examples, config)
        gold = {(item['table_id'], cell['cell_id']) for item in examples for cell in item['evidence_cells']}
        self.assertTrue(all(not gold.intersection(pair['negative_keys']) for pair in pairs))
        self.assertEqual(len(pairs), 2829)

    def test_reference_rejects_wrong_pool_and_gold(self):
        snapshot = read_data('test_candidates.json')
        labels = query_labels('test')
        saved = load_json(HERE / 'reference' / 'test_rankings.json')['records']
        changed = copy.deepcopy(saved)
        changed[0]['ranking'].pop()
        with self.assertRaisesRegex(ValueError, 'pool differs'):
            validate_rankings(changed, snapshot, labels)
        changed = copy.deepcopy(saved)
        changed[0]['gold'] = [['t', 'invented']]
        with self.assertRaisesRegex(ValueError, 'Gold labels differ'):
            validate_rankings(changed, snapshot, labels)

    def test_train_save_reload_and_rank_with_tiny_model(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from table_retrieval.experiments.stage2_bge.train import fit
        from table_retrieval.experiments.stage2_bge.evaluate import load_selector
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / 'vocab.txt'
            vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'a', 'b', '|']))
            tokenizer = BertTokenizerFast(vocab_file=str(vocab), vocab=str(vocab))
            model = BertModel(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=32))
            config = dict(load_json(HERE / 'config.json'), epochs=1, max_length=32)
            examples = read_data('train.json')['annotations'][:2]
            validation = read_data('val.json')['annotations'][:1]
            output = root / 'run'
            result = fit(model, tokenizer, examples, validation, config, output, 'cpu')
            self.assertEqual(len(result['epochs']), 1)
            self.assertTrue(result['checkpoint_sha256'])
            selector = load_selector(output / result['checkpoint'], 'cpu')
            snapshot = read_data('test_candidates.json')
            query = snapshot['queries'][0]
            cells = candidate_cells(query, snapshot)
            selected = selector.rank_prepared_cells('a', cells[:5])
            self.assertEqual(len(selected), 5)
            self.assertEqual(selected[0]['global_rank'], 1)
            with self.assertRaisesRegex(ValueError, 'Output already exists'):
                fit(model, tokenizer, examples, validation, config, output, 'cpu')


if __name__ == '__main__':
    unittest.main()
