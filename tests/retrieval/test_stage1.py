"""Small deterministic tests of the stable Stage 1 serving path."""
import unittest
from unittest.mock import Mock, patch
import numpy as np

from table_retrieval.stage1.fusion import fuse
from table_retrieval.stage1.pipeline import RetrievalPipeline
from table_retrieval.stage1.retrieval import DEFAULT_TABLE_TOP_K, search_query
from table_retrieval.stage1.retrievers.bm25 import BM25
from table_retrieval.stage1.retrievers.ranking import rank_tables
from table_retrieval.stage1.text import table_text


class Stage1Tests(unittest.TestCase):
    def test_equal_weight_rrf_and_id_ties(self):
        actual = fuse([('b', 900), ('a', -1)], [('a', 2), ('b', 1)])
        self.assertEqual([uid for uid, _ in actual], ['a', 'b'])
        self.assertAlmostEqual(actual[0][1], 1 / 61 + 1 / 62)
        self.assertEqual(actual[0][1], actual[1][1])
        self.assertEqual(fuse([('x', 1)], [('y', 2)], depth=1)[0][0], 'x')

    @patch('table_retrieval.stage1.retrievers.bge.search_texts')
    def test_fusion_uses_matched_bm25_and_frozen_depth(self, dense):
        lexical = Mock()
        lexical.search.return_value = [('b', 1), ('a', 2)]
        dense.return_value = ([[('a', .8), ('c', .7)]], [])
        manifest = {'candidate_depth': 100, 'rrf_constant': 60}
        actual = search_query('question', 'fusion', {
            'bm25_matched': (lexical, manifest), 'bge': 'dense-index'})
        self.assertEqual(actual, fuse(lexical.search.return_value, dense.return_value[0][0]))
        lexical.search.assert_called_once_with('question', 100)
        dense.assert_called_once_with('dense-index', ['question'], 100)

    def test_bm25_preserves_decimal_tokens_and_breaks_ties_by_id(self):
        model = BM25({'b': 'accuracy 92.5', 'a': 'accuracy 92.5', 'c': 'unrelated'})
        self.assertEqual([uid for uid, _ in model.search('92.5')], ['a', 'b'])

    def test_dense_ties_preserve_corpus_order(self):
        indices, scores = rank_tables(np.array([[1., 0.]]),
                                      np.array([[1., 0.], [1., 0.], [0., 1.]]), 2)
        self.assertEqual(indices.tolist(), [[0, 1]])
        self.assertEqual(scores.tolist(), [[1., 1.]])

    @patch('table_retrieval.stage1.pipeline.search_query')
    def test_default_five_and_candidate_interface(self, retrieve):
        retrieve.return_value = [('table', .03), ('missing', .02)]
        pipeline = RetrievalPipeline('fusion', {}, {'table': {'rows': [['x']]}})
        candidates = pipeline.search(' question ')
        retrieve.assert_called_once_with('question', 'fusion', {}, 5)
        self.assertEqual(DEFAULT_TABLE_TOP_K, 5)
        self.assertEqual([(c.table_id, c.retrieval_rank, c.retrieval_score) for c in candidates],
                         [('table', 1, .03), ('missing', 2, .02)])
        self.assertIsNone(candidates[1].table)

    def test_stage1_field_selection_does_not_change(self):
        actual = table_text({'paper_title': 'Title', 'caption': 'Caption',
            'sub_caption': 'omit', 'reference_text': 'omit',
            'rows': [['Method', 'Score'], ['A', 0], ['ragged', None, 'x']]})
        self.assertEqual(actual, 'Title\nCaption\nMethod | Score\nA | 0\nragged |  | x')

    def test_invalid_query_or_limit(self):
        pipeline = RetrievalPipeline('fusion', {}, {})
        for query, limit in [(' ', 5), (None, 5), ('q', 0), ('q', True)]:
            with self.subTest(query=query, limit=limit), self.assertRaises(ValueError):
                pipeline.search(query, limit)


if __name__ == '__main__':
    unittest.main()
