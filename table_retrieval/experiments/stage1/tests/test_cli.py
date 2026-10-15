"""The public CLI is small; completed artifacts cannot be silently replaced."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from table_retrieval.experiments.stage1 import cli, config, indexing, workflow
from table_retrieval.stage1 import artifacts
from table_retrieval import data
from table_retrieval.experiments.stage1.data import write_run


class WorkflowTests(unittest.TestCase):
    def test_no_arguments_only_prints_help(self):
        with patch.object(workflow, 'run') as run, redirect_stdout(StringIO()) as output:
            cli.main([])
        run.assert_not_called()
        self.assertIn('--model bge --split val', output.getvalue())

    def test_experiment_routing(self):
        with patch.object(workflow, 'run') as run:
            cli.main(['--model', 'bm25', '--split', 'test', '--experiment', 'trial'])
        self.assertEqual(run.call_args.args[1:], ('bm25', 'test', 'trial'))
        self.assertEqual(run.call_args.kwargs, {'qrels': None})

    def test_qrels_option_routing(self):
        with patch.object(workflow, 'run') as run:
            cli.main(['--model', 'fusion', '--split', 'test', '--qrels', 'custom.tsv'])
        self.assertEqual(run.call_args.kwargs, {'qrels': Path('custom.tsv')})

    def test_qrels_defaults_and_split_specific_override(self):
        settings_path = Path(config.__file__).with_name('settings.json')
        settings = data.load_json(settings_path)
        settings['device'] = 'cpu'
        root = Path(settings['data'])
        self.assertEqual(config.arguments(settings).qrels, root / 'Val_table_qrels.tsv')
        self.assertEqual(config.arguments(settings, True).qrels,
                         root / 'Test_table_qrels.INSTRUCTOR_ONLY.tsv')
        for split in ('val', 'test'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'custom.tsv'
                path.write_text('q1 0 t1 1\n')
                with patch.object(workflow, 'workflow') as run_workflow, \
                        patch.object(workflow.os, 'chdir'):
                    workflow.run(settings_path, 'fusion', split, qrels=path)
                passed = run_workflow.call_args.args[0]
                passed['device'] = 'cpu'
                self.assertEqual(passed[f'{split}_qrels'], str(path.resolve()))
                self.assertEqual(config.arguments(passed, test=split == 'test').qrels, path.resolve())
                other = config.arguments(passed, test=split != 'test').qrels
                self.assertEqual(other, root / ('Val_table_qrels.tsv' if split == 'test'
                                               else 'Test_table_qrels.INSTRUCTOR_ONLY.tsv'))

    def test_internal_stages_are_not_public_commands(self):
        for action in ['index', 'retrieve', 'evaluate', 'run', 'convert', 'train-tapas']:
            with self.subTest(action=action), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                cli.main([action])

    def test_changed_configuration_cannot_reuse_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = data.load_json(Path(config.__file__).with_name('settings.json'))
            settings.update(device='cpu', validation_output=directory)
            args = config.arguments(settings)
            manifest = {key: settings[key] for key in ['model', 'revision', 'max_length', 'candidate_depth']}
            manifest['max_length'] = 1024
            data.write_json(args.output / 'prepared.json', manifest)
            with self.assertRaisesRegex(ValueError, 'differs'):
                config.check_settings(settings, args)

    def test_completed_and_partial_files_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.json'
            data.write_json(path, {'recall@10': .9})
            before = path.stat().st_mtime_ns
            data.write_json(path, {'recall@10': .9})
            self.assertEqual(path.stat().st_mtime_ns, before)
            with self.assertRaisesRegex(ValueError, 'differs'):
                data.write_json(path, {'recall@10': .8})
            run = Path(directory) / 'bm25.run'
            run.with_suffix('.run.tmp').write_text('partial')
            with self.assertRaises(FileExistsError):
                write_run(run, {'q': [('t', 1)]}, 'bm25')
            self.assertEqual(run.with_suffix('.run.tmp').read_text(), 'partial')

    def test_named_output_layout(self):
        settings = data.load_json(Path(config.__file__).with_name('settings.json'))
        settings.update(device='cpu', experiment='trial')
        val, test = config.arguments(settings), config.arguments(settings, True)
        self.assertEqual(val.rankings, Path('outputs/rankings/val/trial'))
        self.assertEqual(test.results, Path('outputs/results/trial/test'))
        self.assertEqual(val.embeddings, test.embeddings)
        self.assertEqual(test.source, val.output)

    def test_explicit_index_corpus_path_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            root = Path(directory)
            index = root / 'portable-index'
            corpus = index / 'assets' / 'Corpus.json'
            corpus.parent.mkdir(parents=True)
            corpus.write_text(json.dumps({'table-1': {'rows': [['A'], ['B']]}}))
            data.write_json(index / 'prepared.json', {
                'corpus': 'assets/Corpus.json',
                'corpus_sha256': data.digest(corpus),
            })
            previous = Path.cwd()
            try:
                os.chdir(elsewhere)
                loaded = artifacts.load_search_corpus(index.resolve())
            finally:
                os.chdir(previous)
            self.assertEqual(loaded['table-1']['rows'][1], ['B'])

    def test_invalid_device_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'Invalid Stage 1 device'):
            artifacts.resolve_device('not-a-real-device')

    def test_historical_code_paths_do_not_block_matching_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / 'embeddings.npy'
            embeddings.write_bytes(b'original embeddings')
            data.write_json(root / 'frozen.json', {
                'code_sha256': {'removed_historical_file.py': 'old hash'},
                'embeddings_sha256': data.digest(embeddings),
            })
            indexing.check_frozen(root)
            embeddings.write_bytes(b'changed embeddings')
            with self.assertRaisesRegex(ValueError, 'new --experiment'):
                indexing.check_frozen(root)


if __name__ == '__main__':
    unittest.main()
