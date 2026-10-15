"""Real encoder smoke tests with a tiny temporary checkpoint, no downloads/training."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import json
import unittest
from unittest.mock import patch
import torch
from transformers import BertTokenizerFast, XLMRobertaConfig, XLMRobertaModel

from table_retrieval.stage2.bge.model import BGECellSelector
from table_retrieval.stage2.bge.pipeline import create_pipeline
from table_retrieval.data import Candidate
from table_retrieval.search import build_pipeline, load_config


class BGETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        cls.previous_rng = torch.random.get_rng_state()
        torch.set_num_threads(1)
        # Stay independent of local workflows that change tempfile's default.
        cls.directory = TemporaryDirectory(prefix='table-bge-test-', dir='/tmp')
        cls.path = Path(cls.directory.name)
        vocab = cls.path / 'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                                  'a', 'accuracy', '92', 'paper', 'table', '|', ':']))
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), vocab=str(vocab))
        tokenizer.save_pretrained(cls.path)
        torch.manual_seed(7)
        model = XLMRobertaModel(XLMRobertaConfig(vocab_size=12, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=1024, pad_token_id=0))
        model.save_pretrained(cls.path)
        cls.max_length = load_config()['stage2_max_length']
        cls.selector = BGECellSelector(str(cls.path), device='cpu', batch_size=2,
                                      max_length=cls.max_length)

    @classmethod
    def tearDownClass(cls):
        del cls.selector
        gc.collect()
        cls.directory.cleanup()
        torch.set_num_threads(cls.previous_threads)
        torch.random.set_rng_state(cls.previous_rng)

    def test_normalized_cls_embeddings_and_checkpoint_compatibility(self):
        vectors = self.selector.encode(['a accuracy 92', 'paper'])
        self.assertEqual(tuple(vectors.shape), (2, 16))
        torch.testing.assert_close(torch.linalg.vector_norm(vectors, dim=1), torch.ones(2))
        self.assertFalse(self.selector.model.training)

    def test_config_token_truncation_local_value_first(self):
        self.assertEqual(self.selector.max_length, self.max_length)
        value = 'a | accuracy | 92 | Table: ' + 'paper ' * 800
        tokenized = self.selector.tokenizer([value], padding=True, truncation=True,
                                           max_length=self.max_length, return_tensors='pt')
        self.assertEqual(tokenized['input_ids'].shape[1], self.max_length)
        self.assertIn('92', self.selector.tokenizer.decode(tokenized['input_ids'][0]))
        encoded = self.selector.encode([value])
        with torch.inference_mode():
            output = self.selector.model(**tokenized).last_hidden_state[:, 0].float()
            expected = torch.nn.functional.normalize(output, dim=-1)
        torch.testing.assert_close(encoded, expected, rtol=0, atol=0)

    def test_configured_token_limit_reaches_encoder_and_provenance(self):
        with TemporaryDirectory(dir='/tmp') as directory:
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps({
                'stage1_index': 'index', 'stage2_checkpoint': str(self.path),
                'stage2_max_length': 128, 'device': 'cpu',
                'table_top_k': 5, 'cell_top_k': 5, 'table_source_dir': None,
            }))
            config = load_config(config_path)
            with patch('table_retrieval.pipeline.RetrievalPipeline.from_index'):
                engine = build_pipeline(config)
        pipeline = engine.cell_retrieval
        selector = pipeline.selector
        self.assertEqual(selector.max_length, 128)
        self.assertEqual(pipeline.provenance['cell_max_length'], 128)
        text = 'paper ' * 800
        inputs = selector.tokenizer([text], padding=True, truncation=True,
                                   max_length=128, return_tensors='pt')
        self.assertEqual(inputs['input_ids'].shape[1], 128)
        encoded = selector.encode([text])
        with torch.inference_mode():
            output = selector.model(**inputs).last_hidden_state[:, 0].float()
            expected = torch.nn.functional.normalize(output, dim=-1)
        torch.testing.assert_close(encoded, expected, rtol=0, atol=0)

    def test_max_length_must_be_a_positive_integer(self):
        for value in (0, -1, 1.5, '512', None, True, False):
            with self.subTest(max_length=value), self.assertRaisesRegex(
                    ValueError, 'max length must be a positive integer'):
                BGECellSelector(str(self.path), device='cpu', max_length=value)

    def test_token_limit_is_required_by_selector_and_factories(self):
        from table_retrieval.pipeline import EvidencePipeline
        with self.assertRaises(TypeError):
            BGECellSelector(str(self.path), device='cpu')
        with self.assertRaises(TypeError):
            create_pipeline(self.path, device='cpu')
        with self.assertRaises(TypeError):
            EvidencePipeline.from_artifacts('index', self.path, device='cpu')

    def test_training_encoder_uses_explicit_token_limit(self):
        from table_retrieval.experiments.stage2_bge.train import encode
        text = 'paper ' * 800
        with patch.object(self.selector, 'tokenizer', wraps=self.selector.tokenizer) as tokenizer:
            with torch.inference_mode():
                vectors = encode(self.selector.model, tokenizer, [text], 'cpu', 64)
        self.assertEqual(tokenizer.call_args.kwargs['max_length'], 64)
        inputs = self.selector.tokenizer([text], padding=True, truncation=True,
                                        max_length=64, return_tensors='pt')
        self.assertEqual(inputs['input_ids'].shape[1], 64)
        with torch.inference_mode():
            output = self.selector.model(**inputs).last_hidden_state[:, 0].float()
            expected = torch.nn.functional.normalize(output, dim=-1)
        torch.testing.assert_close(vectors, expected, rtol=0, atol=0)

    def test_stage2_cli_reads_runtime_settings_from_config(self):
        from table_retrieval.stage2.cli import main
        with TemporaryDirectory(dir='/tmp') as directory:
            root = Path(directory)
            config = dict(load_config(), stage2_checkpoint=str(self.path), device='cpu',
                          stage2_max_length=64, cell_top_k=1)
            (root / 'config.json').write_text(json.dumps(config))
            (root / 'candidates.json').write_text(json.dumps([
                {'table_id': 'table', 'table': {'rows': [['accuracy', '92']]}}
            ]))
            with patch('builtins.print'):
                main(['--query', 'accuracy', '--config', str(root / 'config.json'),
                      '--candidates', str(root / 'candidates.json'),
                      '--output', str(root / 'predictions.jsonl')])
            result = json.loads((root / 'predictions.jsonl').read_text())
        self.assertEqual(result['configuration']['cell_max_length'], 64)
        self.assertEqual(result['configuration']['cell_model'], str(self.path.resolve()))
        self.assertEqual(result['configuration']['cell_device'], 'cpu')
        self.assertEqual(result['configuration']['cell_top_k'], 1)
        self.assertEqual(len(result['cells']), 1)
        self.assertEqual(result['status'], 'complete')

    def test_identical_text_scores_keep_prepared_order(self):
        cells = [dict(text='a accuracy 92', cell_id=uid) for uid in ['b', 'a']]
        ranked = self.selector.rank_prepared_cells('accuracy', cells)
        self.assertEqual([cell['cell_id'] for cell in ranked], ['b', 'a'])
        self.assertEqual([cell['global_rank'] for cell in ranked], [1, 2])

    def test_factory_global_five_and_no_checkpoint_writes(self):
        before = {path.name: path.read_bytes() for path in self.path.iterdir() if path.is_file()}
        pipeline = create_pipeline(self.path, device='cpu', max_length=self.max_length)
        self.assertEqual(pipeline.provenance['cell_max_length'], self.max_length)
        result = pipeline.rank('accuracy', [Candidate('table', {
            'rows': [['Model', 'Accuracy'], ['A', '92'], ['B', '85']], 'paper_title': 'Paper'})],
                               load_config()['cell_top_k'])
        self.assertEqual(len(result['cells']), 5)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(pipeline.provenance['cell_representation'], 'structural_v1')
        after = {path.name: path.read_bytes() for path in self.path.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_encoder_input_validation(self):
        for query, cells in [(' ', [{'text': 'x'}]), ('q', [{}])]:
            with self.assertRaises(ValueError):
                self.selector.rank_prepared_cells(query, cells)
        self.assertEqual(self.selector.rank_prepared_cells('q', []), [])
        with self.assertRaises(FileNotFoundError):
            create_pipeline(self.path / 'missing', max_length=self.max_length)


if __name__ == '__main__':
    unittest.main()
