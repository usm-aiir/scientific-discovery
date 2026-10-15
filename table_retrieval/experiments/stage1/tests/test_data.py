"""Regression coverage for relocated evaluation formats and shared ID validation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from table_retrieval import data
from table_retrieval.experiments.stage1.data import load_qrels, load_queries, read_run, write_run
from table_retrieval.experiments.stage1.metrics import evaluate


class EvaluationDataTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(dir='/tmp')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_run_format_preserves_ids_order_and_score_precision(self):
        path = self.root / 'rankings.run'
        rankings = {'q2': [('2504.12345::T2::a', 1 / 3), ('table', -2)],
                    'q1': [('other', 1.23456789012345)]}
        write_run(path, rankings, 'bge')
        self.assertEqual(path.read_text(),
                         'q2 Q0 2504.12345::T2::a 1 0.333333333333 bge\n'
                         'q2 Q0 table 2 -2 bge\n'
                         'q1 Q0 other 1 1.23456789012 bge\n')
        self.assertEqual(read_run(path), {
            'q2': [('2504.12345::T2::a', .333333333333), ('table', -2.0)],
            'q1': [('other', 1.23456789012)]})
        self.assertFalse(path.with_suffix('.run.tmp').exists())

    def test_run_write_protects_existing_and_partial_files(self):
        path = self.root / 'rankings.run'
        path.write_text('existing run')
        with self.assertRaisesRegex(ValueError, 'Existing rankings'):
            write_run(path, {'q': [('table', 1)]}, 'bge')
        self.assertEqual(path.read_text(), 'existing run')
        other = self.root / 'partial.run'
        partial = other.with_suffix('.run.tmp')
        partial.write_text('partial run')
        with self.assertRaises(FileExistsError):
            write_run(other, {'q': [('table', 1)]}, 'bge')
        self.assertEqual(partial.read_text(), 'partial run')
        self.assertFalse(other.exists())

    def test_run_read_sorts_saved_ranks_without_normalizing_ids(self):
        path = self.root / 'rankings.run'
        path.write_text('q Q0 second 2 0.2 bge\nq Q0 2504.12345::T2:: 1 0.9 bge\n')
        self.assertEqual(read_run(path), {'q': [('2504.12345::T2::', .9), ('second', .2)]})

    def test_queries_preserve_input_ids_text_and_extra_fields(self):
        path = self.root / 'queries.json'
        queries = [{'query_id': 7, 'query': ' accuracy? ', 'source_table_uids': ['table']}]
        path.write_text(json.dumps(queries))
        self.assertEqual(load_queries(path), queries)

    def test_queries_keep_existing_validation(self):
        path = self.root / 'queries.json'
        invalid = [[], {}, [{}], [{'query_id': 'q', 'query': ' '}],
                   [{'query_id': 'q', 'query': None}], [{'query_id': 'q 1', 'query': 'text'}],
                   [{'query_id': 1, 'query': 'first'}, {'query_id': '1', 'query': 'second'}]]
        for value in invalid:
            with self.subTest(queries=value):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_queries(path)

    def test_qrels_keep_grade_filtering_and_id_normalization(self):
        path = self.root / 'qrels.tsv'
        path.write_text('\nq1 0 2504.12345::T2::a 2\n'
                        'q1 0 2504.12345_2_a 1\n'
                        'q1 0 2504.12345v2::T3:: 1\n'
                        'q2 0 unchanged 1\nq3 0 ignored 0\n')
        self.assertEqual(dict(load_qrels(path)), {
            'q1': {'2504.12345_2_a', '2504.12345v2_3_'}, 'q2': {'unchanged'}})
        self.assertEqual(dict(load_qrels(path, min_grade=2)), {'q1': {'2504.12345_2_a'}})
        for text in ('q 0 table\n', 'q 0 table invalid-grade\n'):
            with self.subTest(qrels=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_qrels(path)

    def test_shared_corpus_loader_still_rejects_normalized_id_collisions(self):
        path = self.root / 'Corpus.json'
        corpus = {'2504.12345::T2::a': {'rows': [['A', 0]]}}
        path.write_text(json.dumps(corpus))
        self.assertEqual(data.load_corpus(path), corpus)
        corpus['2504.12345_2_a'] = {'rows': [['B', 1]]}
        path.write_text(json.dumps(corpus))
        with self.assertRaisesRegex(ValueError, 'Ambiguous corpus table IDs'):
            data.load_corpus(path)

    def test_evaluation_still_matches_raw_run_ids_to_normalized_qrels(self):
        run = self.root / 'rankings.run'
        qrels = self.root / 'qrels.tsv'
        write_run(run, {'q1': [('2504.12345::T2::a', .9)], 'q2': [('other', .8)]}, 'bge')
        qrels.write_text('q1 0 2504.12345_2_a 1\nq2 0 missing 2\n')
        result = evaluate(run, qrels, ['q1', 'q2'])
        self.assertEqual(result['recall@10'], .5)
        self.assertEqual(result['per_query'], {'q1': 1.0, 'q2': 0.0})

    def test_offline_workflow_reuses_runs_and_freezes_relocated_code(self):
        import torch
        from transformers import BertTokenizerFast, XLMRobertaConfig, XLMRobertaModel
        from table_retrieval.experiments.stage1.workflow import run

        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        model_path = self.root / 'model'
        model_path.mkdir()
        vocab = model_path / 'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                                    'alpha', 'beta', 'value', '0', '1', '|']))
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), vocab=str(vocab))
        tokenizer.save_pretrained(model_path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = XLMRobertaModel(XLMRobertaConfig(
                vocab_size=11, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, pad_token_id=0))
            model.save_pretrained(model_path)
        dataset = self.root / 'dataset'
        dataset.mkdir()
        (dataset / 'Corpus.json').write_text(json.dumps({
            'a': {'paper_title': 'alpha', 'rows': [['value'], ['0']]},
            'b': {'paper_title': 'beta', 'rows': [['value'], ['1']]}}))
        (dataset / 'Val.json').write_text(json.dumps([
            {'query_id': 'q1', 'query': 'alpha'}, {'query_id': 'q2', 'query': 'beta'}]))
        (dataset / 'Val_table_qrels.tsv').write_text('q1 0 a 1\nq2 0 b 1\n')
        (dataset / 'Test.json').write_text(json.dumps([
            {'query_id': 'q3', 'query': 'alpha value'}, {'query_id': 'q4', 'query': 'beta value'}]))
        (dataset / 'Test_table_qrels.INSTRUCTOR_ONLY.tsv').write_text('q3 0 a 1\nq4 0 b 1\n')
        settings = self.root / 'settings.json'
        settings.write_text(json.dumps({
            'device': 'cpu', 'batch_size': 2, 'data': str(dataset),
            'validation_output': str(self.root / 'legacy-val'),
            'test_output': str(self.root / 'legacy-test'),
            'model': str(model_path), 'revision': 'main', 'max_length': 32,
            'candidate_depth': 100, 'top_k': 10, 'output_root': str(self.root / 'outputs')}))
        with patch('table_retrieval.experiments.stage1.workflow.os.chdir'):
            run(settings, 'fusion', 'val', 'smoke')
            run(settings, 'fusion', 'test', 'smoke')
            runs = self.root / 'outputs' / 'rankings' / 'val' / 'smoke'
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in runs.glob('*.run')}
            with patch('table_retrieval.experiments.stage1.retrievers.bge.index_bge') as rebuild:
                run(settings, 'fusion', 'val', 'smoke')
            rebuild.assert_not_called()
        self.assertEqual(len(before), 4)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before})
        for split in ('val', 'test'):
            metrics = data.load_json(self.root / 'outputs' / 'results' / 'smoke' / split / 'fusion.metrics.json')
            self.assertEqual(metrics['recall@10'], 1.0)
        index = self.root / 'outputs' / 'indexes' / 'smoke' / 'test'
        self.assertTrue((index / 'embeddings.npy').is_symlink())
        frozen = data.load_json(index / 'frozen.json')
        self.assertEqual(frozen['code_sha256']['experiments/stage1/data.py'],
                         data.digest(Path(__file__).resolve().parents[1] / 'data.py'))


if __name__ == '__main__':
    unittest.main()
