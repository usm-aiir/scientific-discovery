"""Stage 2 CLI configuration routing and existing failure contracts."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from table_retrieval.data import Candidate
from table_retrieval.search import load_config
from table_retrieval.stage2 import cli


class Stage2CLITests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(dir='/tmp')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / 'runtime.json'
        self.config = dict(load_config(), stage2_checkpoint='checkpoint', device='cpu',
                           table_source_dir='html', cell_top_k=3, stage2_max_length=96)
        self.config_path.write_text(json.dumps(self.config))
        self.candidates_path = self.root / 'candidates.json'
        self.candidate = Candidate('table', {'rows': [['Accuracy', '92']]}, 1, .3)
        self.candidates_path.write_text(json.dumps([self.candidate.to_dict()]))
        self.output = self.root / 'predictions.jsonl'
        self.argv = ['--query', 'accuracy', '--query-id', 'q1',
                     '--candidates', str(self.candidates_path), '--output', str(self.output)]
        self.engine = Mock(provenance={'cell_max_length': 96})
        self.engine.rank.return_value = {
            'status': 'complete', 'tables': [], 'cells': [], 'errors': []}
        factory = patch.object(cli, 'create_pipeline', return_value=self.engine)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        stderr = patch('sys.stderr', new_callable=io.StringIO)
        self.stderr = stderr.start()
        self.addCleanup(stderr.stop)

    def run_cli(self, extra=()):
        with patch('builtins.print'):
            cli.main([*self.argv, '--config', str(self.config_path), *extra])

    def test_config_routes_all_runtime_settings_and_resolves_relative_paths(self):
        self.run_cli()
        self.factory.assert_called_once_with(
            str(self.root / 'checkpoint'), 'cpu', max_length=96,
            table_source_dir=str(self.root / 'html'))
        self.engine.rank.assert_called_once_with('accuracy', [self.candidate], 3)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['query_id'], 'q1')
        self.assertEqual(result['configuration']['cell_top_k'], 3)
        self.assertEqual(result['configuration']['cell_max_length'], 96)
        self.assertEqual(result['configuration']['table_top_k'], 1)

    def test_default_runtime_config_is_used_when_config_argument_is_omitted(self):
        with patch.object(cli, 'load_config', wraps=load_config) as load, patch('builtins.print'):
            cli.main(self.argv)
        load.assert_called_once_with(cli.DEFAULT_CONFIG)
        self.engine.rank.assert_called_once_with(
            'accuracy', [self.candidate], load_config()['cell_top_k'])

    def test_removed_runtime_flags_are_rejected(self):
        for flag, value in (('--checkpoint', 'other'), ('--cell-device', 'cpu'),
                            ('--table-source-dir', 'other'), ('--cell-top-k', '1')):
            with self.subTest(flag=flag), self.assertRaises(SystemExit) as raised:
                self.run_cli([flag, value])
            self.assertEqual(raised.exception.code, 2)
        self.factory.assert_not_called()

    def test_invalid_runtime_values_fail_before_pipeline_creation(self):
        for key, value in (('cell_top_k', 0), ('cell_top_k', True),
                           ('stage2_max_length', -1), ('device', '')):
            with self.subTest(key=key, value=value):
                self.config_path.write_text(json.dumps(dict(self.config, **{key: value})))
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli()
                self.assertEqual(raised.exception.code, 1)
        self.factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_empty_query_and_invalid_candidate_list_are_rejected(self):
        self.argv[1] = ' '
        with self.assertRaises(SystemExit) as raised:
            self.run_cli()
        self.assertEqual(raised.exception.code, 2)
        self.argv[1] = 'accuracy'
        self.candidates_path.write_text('{}')
        with self.assertRaises(SystemExit) as raised:
            self.run_cli()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('Candidates must be a JSON list', self.stderr.getvalue())
        self.factory.assert_not_called()

    def test_existing_output_or_partial_file_is_not_overwritten(self):
        for path in (self.output, self.output.with_name(self.output.name + '.partial')):
            with self.subTest(path=path.name):
                path.write_text('existing output')
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli()
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(path.read_text(), 'existing output')
                path.unlink()
        self.factory.assert_not_called()

    def test_checkpoint_loading_error_keeps_existing_cli_error_handling(self):
        self.factory.side_effect = FileNotFoundError('checkpoint missing')
        with self.assertRaises(SystemExit) as raised:
            self.run_cli()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('Cell selection failed: checkpoint missing', self.stderr.getvalue())
        self.engine.rank.assert_not_called()

    def test_partial_result_is_saved_before_exit_code_two(self):
        self.engine.rank.return_value.update(status='partial', errors=[{'error': 'bad table'}])
        with self.assertRaises(SystemExit) as raised:
            self.run_cli()
        self.assertEqual(raised.exception.code, 2)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['errors'], [{'error': 'bad table'}])


if __name__ == '__main__':
    unittest.main()
