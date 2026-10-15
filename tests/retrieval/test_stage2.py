"""Structural text, source alignment, cell identity and global ranking contracts."""
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import torch

from table_retrieval.data import Candidate
from table_retrieval.pipeline import EvidencePipeline
from table_retrieval.search import build_pipeline, load_config, main as search_cli
from table_retrieval.stage2.cells import make_cells
from table_retrieval.stage2.pipeline import BGECellRetrievalPipeline
from table_retrieval.stage2.ranking import rank_scored_cells
from table_retrieval.stage2.representation import build_structural_cells, structural_bge_text
from table_retrieval.stage2.source import source_tables, load_table_sources
from table_retrieval.stage2.structure import find_alignment

HTML = '''<table id="T1"><tr><th rowspan="2">Model</th><th colspan="2">Dataset X</th></tr>
<tr><th>Accuracy</th><th>F1</th></tr><tr><td>A</td><td>92</td><td>90</td></tr>
<tr><td>B</td><td>85</td><td>84</td></tr></table>'''
TABLE = {'rows': [['Model', 'Dataset X'], ['Accuracy', 'F1'], ['A', '92', '90'], ['B', '85', '84']],
         'caption': 'Results.', 'sub_caption': 'Panel A', 'paper_title': 'Paper'}


class RecordingSelector:
    def rank_prepared_cells(self, query, cells):
        self.query, self.cells = query, deepcopy(cells)
        return rank_scored_cells(cells, torch.zeros(len(cells)))


class Stage2Tests(unittest.TestCase):
    def test_end_to_end_smoke_returns_five_tables_and_global_five_cells(self):
        retrieval = Mock(retriever='fusion', provenance={})
        retrieval.search.return_value = [
            Candidate(f'table-{index}', TABLE, index, .04 - index / 100)
            for index in range(1, 6)
        ]
        cells = BGECellRetrievalPipeline(RecordingSelector())
        engine = EvidencePipeline(retrieval, cells)
        config = load_config()
        result = engine.search('question', config['table_top_k'], cell_top_k=config['cell_top_k'])
        retrieval.search.assert_called_once_with('question', 5)
        self.assertEqual(len(result['tables']), 5)
        self.assertEqual(len(result['cells']), 5)
        self.assertEqual(result['configuration']['cell_top_k'], 5)
        self.assertEqual(result['configuration']['table_top_k'], 5)
        self.assertEqual(result['configuration']['cell_selection'], 'global_top_k')
        self.assertNotIn('aggregation', result)
        self.assertNotIn('evidence_groups', result)

    def test_combined_factory_forwards_source_and_configured_token_limit(self):
        max_length = load_config()['stage2_max_length']
        with patch('table_retrieval.pipeline.RetrievalPipeline.from_index') as tables, \
                patch('table_retrieval.pipeline.create_pipeline') as cells:
            engine = EvidencePipeline.from_artifacts(
                stage1_index='index', checkpoint='checkpoint',
                device='cpu', table_source_dir='html', stage2_max_length=max_length)
        self.assertIs(engine.retrieval, tables.return_value)
        self.assertIs(engine.cell_retrieval, cells.return_value)
        tables.assert_called_once_with('index', 'cpu')
        cells.assert_called_once_with('checkpoint', 'cpu', max_length=max_length, table_source_dir='html')

    def test_live_search_uses_configured_global_cell_limit(self):
        retrieval = Mock(provenance={})
        retrieval.search.return_value = [Candidate('a', TABLE), Candidate('b', TABLE)]
        cells = BGECellRetrievalPipeline(RecordingSelector())
        engine = EvidencePipeline(retrieval, cells)
        with TemporaryDirectory(dir='/tmp') as directory:
            path = Path(directory) / 'runtime.json'
            path.write_text(json.dumps(dict(load_config(), table_top_k=2, cell_top_k=3)))
            with patch('table_retrieval.search.build_pipeline', return_value=engine), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                search_cli(['question', '--config', str(path)])
        result = json.loads(output.getvalue())
        retrieval.search.assert_called_once_with('question', 2)
        self.assertEqual(result['configuration']['table_top_k'], 2)
        self.assertEqual(result['configuration']['cell_top_k'], 3)
        self.assertEqual(len(result['cells']), 3)
        self.assertTrue(all(cell['table_id'] == 'a' for cell in result['cells']))

    def test_cell_limit_is_required_and_must_be_a_positive_integer(self):
        cells = BGECellRetrievalPipeline(RecordingSelector())
        with self.assertRaises(TypeError):
            cells.rank('query', [])
        retrieval = Mock()
        with self.assertRaises(TypeError):
            EvidencePipeline(retrieval, cells).search('query', load_config()['table_top_k'])
        retrieval.search.assert_not_called()
        for value in (0, -1, True, False, 1.5, '5', None):
            with self.subTest(cell_top_k=value), self.assertRaisesRegex(
                    ValueError, 'cell_top_k must be a positive integer'):
                cells.rank('query', [], value)

    def test_table_limit_is_required_even_when_cell_limit_is_provided(self):
        retrieval = Mock()
        engine = EvidencePipeline(retrieval, Mock())
        with self.assertRaisesRegex(TypeError, 'table_top_k'):
            engine.search('query', cell_top_k=load_config()['cell_top_k'])
        retrieval.search.assert_not_called()
        engine.cell_retrieval.rank.assert_not_called()

    def test_config_requires_a_positive_integer_stage2_max_length(self):
        values = load_config()
        self.assertEqual(values['stage2_max_length'], 512)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for value in (0, -1, 1.5, '512', None, True, False):
                with self.subTest(stage2_max_length=value):
                    path.write_text(json.dumps(dict(values, stage2_max_length=value)))
                    with self.assertRaisesRegex(ValueError, 'stage2_max_length must be a positive integer'):
                        load_config(path)

    def test_exact_text_format_and_optional_context_order(self):
        cell = {'row_label': 'A', 'value': 0}
        context = {'header_path': ['Dataset X', 'Accuracy'], 'row_context': ['Model=A']}
        actual = structural_bge_text(cell, TABLE, context)
        self.assertEqual(actual, 'A | Dataset X > Accuracy | 0 | Row context: Model=A'
                         ' | Table: Results. | Subtable: Panel A | Paper: Paper')
        self.assertEqual(structural_bge_text(cell, {}, dict(context, row_context=[])),
                         'A | Dataset X > Accuracy | 0')

    def test_source_spans_are_used_without_changing_coordinates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / '2504.12345.html'
            path.write_text(HTML)
            cells, audit = build_structural_cells('2504.12345::T1', TABLE, sources=source_tables(path))
        value = next(cell for cell in cells if cell['cell_id'] == 'r2:c1')
        self.assertEqual((value['row'], value['column'], value['value']), (2, 1, '92'))
        self.assertEqual(value['text'], 'A | Dataset X > Accuracy | 92 | Row context: Model=A'
                         ' | Table: Results. | Subtable: Panel A | Paper: Paper')
        self.assertEqual(audit['alignment'], 'exact_html_row_alignment')
        self.assertNotIn('90', value['text'])

    def test_ambiguous_and_nonexact_alignment_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.html'
            path.write_text(HTML + HTML)
            sources = source_tables(path)
        self.assertEqual(find_alignment(TABLE['rows'], sources), (None, 'ambiguous_alignment'))
        different = deepcopy(TABLE['rows'])
        different[2][1] = '92.0'
        self.assertEqual(find_alignment(different, sources), (None, 'no_exact_alignment'))

    def test_source_math_footnotes_and_nested_headers(self):
        html = '''<table><tr><th><table><tr><td>Multi</td><td>line</td></tr></table></th>
        <td><math alttext="x_1">ignored</math><span class="ltx_note ltx_role_footnote">
        <sup class="ltx_note_mark">2</sup><span class="ltx_note_content">note</span></span></td>
        </tr></table>'''
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.html'
            path.write_text(html)
            sources = source_tables(path)
        self.assertEqual(len(sources), 1)
        self.assertEqual([cell['text'] for cell in sources[0]['cells']], ['Multi line', '$x_1$[^2]'])

    def test_embedded_spans_require_the_same_exact_alignment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.html'
            path.write_text(HTML)
            sources = source_tables(path)
        embedded = dict(TABLE, cells=sources[0]['cells'], n_cols=3)
        cells, audit = build_structural_cells('table', embedded)
        self.assertEqual(audit['alignment'], 'exact_html_row_alignment')
        cell = next(cell for cell in cells if cell['cell_id'] == 'r2:c2')
        self.assertIn('Dataset X > F1 | 90', cell['text'])
        embedded['rows'] = deepcopy(TABLE['rows'])
        embedded['rows'][2][2] = '90.0'
        _, audit = build_structural_cells('table', embedded)
        self.assertEqual(audit['alignment'], 'no_exact_alignment')

    def test_legacy_zero_span_positions_are_retained(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.html'
            path.write_text('<table><tr><td rowspan="0">A</td><td colspan="0">B</td><td>C</td></tr></table>')
            cells = source_tables(path)[0]['cells']
        self.assertEqual([(cell['row'], cell['col']) for cell in cells], [(0, 0), (0, 1), (0, 1)])

    def test_ragged_indicators_recover_only_supported_leaf_headers(self):
        table = {'rows': [['Feature A', 'Feature B', 'Results'], ['Accuracy', 'F1'],
                          ['✓', '', '92', '90'], ['', '✓', '85', '84']]}
        cells, audit = build_structural_cells('table', table)
        cell = next(cell for cell in cells if cell['cell_id'] == 'r2:c2')
        self.assertEqual(cell['text'], '✓ | Accuracy | 92 | Row context: Feature A=✓; Feature B=<empty>')
        self.assertNotIn('90', cell['text'])
        self.assertEqual(audit['alignment'], 'html_unavailable')

    def test_unknown_ragged_headers_are_not_fabricated(self):
        table = {'rows': [['Group A', 'Group B'], ['metric'], ['A', '1', '2', '3']]}
        cells, audit = build_structural_cells('table', table)
        self.assertEqual(cells[-1]['text'], 'A | Column 3 | 3')
        self.assertEqual(audit['body_start'], None)

    def test_legacy_make_cells_preserves_blanks_zero_and_exact_text(self):
        table = {'rows': [['Method', 'Value'], ['A', 0], ['', None], ['x']]}
        cells = make_cells('id', table)
        self.assertEqual([cell['cell_id'] for cell in cells], ['r0:c0', 'r0:c1', 'r1:c0', 'r1:c1', 'r3:c0'])
        self.assertEqual(cells[3]['text'], 'A | Value | 0')

    def test_structural_build_does_not_mutate_table_or_legacy_metadata(self):
        table = deepcopy(TABLE)
        before = deepcopy(table)
        legacy = make_cells('table', table)
        structural, _ = build_structural_cells('table', table)
        self.assertEqual(table, before)
        for old, new in zip(legacy, structural):
            self.assertEqual({k: v for k, v in old.items() if k != 'text'},
                             {k: v for k, v in new.items() if k != 'text'})

    def test_global_five_keeps_candidate_order_for_ties_and_pools_all_cells(self):
        selector = RecordingSelector()
        pipeline = BGECellRetrievalPipeline(selector)
        candidates = [Candidate('b', TABLE, 1, .03), Candidate('a', TABLE, 2, .02)]
        cell_top_k = load_config()['cell_top_k']
        result = pipeline.rank(' query ', candidates, cell_top_k)
        self.assertEqual(cell_top_k, 5)
        self.assertEqual(len(result['cells']), 5)
        self.assertEqual(len(selector.cells), len(make_cells('b', TABLE)) * 2)
        self.assertEqual(selector.query, 'query')
        self.assertTrue(all(cell['table_id'] == 'b' for cell in result['cells']))
        self.assertEqual([cell['global_rank'] for cell in result['cells']], list(range(1, 6)))
        self.assertTrue(all(cell['stage1_table_score'] == .03 for cell in result['cells']))

    def test_bad_candidate_is_partial_and_empty_candidates_are_error(self):
        pipeline = BGECellRetrievalPipeline(RecordingSelector())
        cell_top_k = load_config()['cell_top_k']
        result = pipeline.rank('query', [Candidate('missing', None), Candidate('valid', TABLE)], cell_top_k)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(pipeline.rank('query', [], cell_top_k)['status'], 'error')

    def test_malformed_optional_html_falls_back_instead_of_losing_cells(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / '2504.12345.html').write_text('<table><tr><td colspan="bad">x</td></tr></table>')
            pipeline = BGECellRetrievalPipeline(RecordingSelector(), table_source_dir=directory)
            result = pipeline.rank('query', [Candidate('2504.12345::T1', TABLE)],
                                   load_config()['cell_top_k'])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['cells']), 5)
        self.assertIn('warnings', result['tables'][0])

    def test_null_or_missing_paper_html_uses_corpus_structure(self):
        table = {'rows': [['Model', 'Accuracy'], ['A', '92'], ['B', '85']],
                 'caption': 'Corpus results'}
        table_id = '2504.12345::T1'
        results = []
        with TemporaryDirectory(dir='/tmp') as directory:
            root = Path(directory)
            (root / 'checkpoint').mkdir()
            (root / 'html').mkdir()
            # Other papers' HTML must not be used for this table.
            (root / 'html' / '2504.54321.html').write_text(HTML)
            for source_dir in (None, 'html'):
                with self.subTest(table_source_dir=source_dir):
                    path = root / 'runtime.json'
                    path.write_text(json.dumps(dict(
                        load_config(), stage2_checkpoint='checkpoint', device='cpu',
                        table_source_dir=source_dir)))
                    config = load_config(path)
                    if source_dir is None:
                        self.assertIsNone(config['table_source_dir'])
                    self.assertEqual(load_table_sources(table_id, config['table_source_dir']),
                                     ((), None))
                    selector = RecordingSelector()
                    selector.revision, selector.device = None, 'cpu'
                    selector.max_length = config['stage2_max_length']
                    retrieval = Mock(provenance={})
                    retrieval.search.return_value = [Candidate(table_id, table)]
                    with patch('table_retrieval.pipeline.RetrievalPipeline.from_index',
                               return_value=retrieval), \
                            patch('table_retrieval.stage2.bge.pipeline.BGECellSelector',
                                  return_value=selector), \
                            patch('table_retrieval.stage2.source.source_tables') as parse_html:
                        engine = build_pipeline(config)
                        result = engine.search('accuracy', config['table_top_k'],
                                               cell_top_k=config['cell_top_k'])
                    parse_html.assert_not_called()
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(len(result['cells']), config['cell_top_k'])
                    self.assertEqual(result['errors'], [])
                    summary = result['tables'][0]
                    self.assertEqual(summary['structure']['alignment'], 'html_unavailable')
                    self.assertIsNone(summary['structure']['html_path'])
                    self.assertNotIn('warnings', summary)
                    cell = next(cell for cell in selector.cells if cell['cell_id'] == 'r1:c1')
                    self.assertEqual(cell['text'], 'A | Accuracy | 92 | Table: Corpus results')
                    results.append((result['cells'], result['tables'], selector.cells))
        self.assertEqual(results[0], results[1])

    def test_missing_source_directory_and_files(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(load_table_sources('../outside', directory), ((), None))
            self.assertEqual(load_table_sources('2504.12345::T1', directory), ((), None))
            with self.assertRaises(FileNotFoundError):
                BGECellRetrievalPipeline(RecordingSelector(), table_source_dir=Path(directory) / 'missing')

    def test_source_cache_invalidates_when_file_changes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / '2504.12345.html'
            path.write_text(HTML)
            first, _ = load_table_sources('2504.12345::T1', directory)
            path.write_text(HTML.replace('Dataset X', 'Dataset Different'))
            second, _ = load_table_sources('2504.12345::T1', directory)
        self.assertEqual(first[0]['cells'][1]['text'], 'Dataset X')
        self.assertEqual(second[0]['cells'][1]['text'], 'Dataset Different')

    def test_nonfinite_or_misaligned_scores_are_errors(self):
        for scores in (torch.tensor([float('nan')]), torch.tensor([]), torch.zeros(1, 1)):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                rank_scored_cells([{'text': 'x'}], scores)

    def test_prepared_scoring_does_not_load_gold_fields(self):
        table = dict(TABLE, evidence_cells=['gold'], explanation='do not embed', query='do not embed')
        cells, _ = build_structural_cells('table', table)
        self.assertTrue(all('do not embed' not in cell['text'] and 'gold' not in cell['text'] for cell in cells))


if __name__ == '__main__':
    unittest.main()
