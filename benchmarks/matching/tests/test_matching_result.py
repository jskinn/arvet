#Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import pickle
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.matching.matching_result as match_res


class TestMatchBenchmarkResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return match_res.MatchBenchmarkResult

    def make_instance(self, *args, **kwargs):
        match_types = [match_res.MatchType.TRUE_POSITIVE, match_res.MatchType.FALSE_POSITIVE,
                       match_res.MatchType.TRUE_POSITIVE, match_res.MatchType.FALSE_NEGATIVE]
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_id': np.random.randint(10, 20),
            'matches': {np.random.uniform(0, 600): match_types[np.random.randint(0, len(match_types))]
                        for _ in range(100)},
            'settings': {}
        })
        return match_res.MatchBenchmarkResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: MatchBenchmarkResult
        :param benchmark_result2: MatchBenchmarkResult
        :return:
        """
        if (not isinstance(benchmark_result1, match_res.MatchBenchmarkResult) or
                not isinstance(benchmark_result2, match_res.MatchBenchmarkResult)):
            self.fail('object was not a MatchBenchmarkResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertEqual(benchmark_result1.matches, benchmark_result2.matches)
        self.assertEqual(benchmark_result1.true_positives, benchmark_result2.true_positives)
        self.assertEqual(benchmark_result1.false_positives, benchmark_result2.false_positives)
        self.assertEqual(benchmark_result1.true_negatives, benchmark_result2.true_negatives)
        self.assertEqual(benchmark_result1.false_negatives, benchmark_result2.false_negatives)
        self.assertEqual(benchmark_result1.precision, benchmark_result2.precision)
        self.assertEqual(benchmark_result1.recall, benchmark_result2.recall)
        self.assertEqual(benchmark_result1.f1, benchmark_result2.f1)
        self.assertEqual(benchmark_result1.settings, benchmark_result2.settings)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'matches':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for BSON
        trans_error1 = pickle.loads(s_model1['matches'])
        trans_error2 = pickle.loads(s_model2['matches'])
        self.assertEqual(set(trans_error1.keys()), set(trans_error2.keys()))
        for key in trans_error1:
            self.assertTrue(np.array_equal(trans_error1[key], trans_error2[key]))

    def test_true_positives_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(list(subject.matches.values()).count(match_res.MatchType.TRUE_POSITIVE),
                         subject.true_positives)

    def test_false_positives_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(list(subject.matches.values()).count(match_res.MatchType.FALSE_POSITIVE),
                         subject.false_positives)

    def test_true_negatives_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(list(subject.matches.values()).count(match_res.MatchType.TRUE_NEGATIVE),
                         subject.true_negatives)

    def test_false_negatives_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(list(subject.matches.values()).count(match_res.MatchType.FALSE_NEGATIVE),
                         subject.false_negatives)
