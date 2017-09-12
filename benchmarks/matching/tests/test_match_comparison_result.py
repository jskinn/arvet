#Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import pickle
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.matching.match_comparison_result as comp_res


class TestMatchingComparisonResult(entity_test.EntityContract, unittest.TestCase):
    def get_class(self):
        return comp_res.MatchingComparisonResult

    def make_instance(self, *args, **kwargs):
        match_types = [comp_res.MatchChanges.LOST_TO_TRUE_POSITIVE,
                       comp_res.MatchChanges.TRUE_POSITIVE_TO_LOST,
                       comp_res.MatchChanges.REMAIN_TRUE_POSITIVE,
                       comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_POSITIVE,
                       comp_res.MatchChanges.TRUE_POSITIVE_TO_TRUE_NEGATIVE,
                       comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_NEGATIVE,
                       comp_res.MatchChanges.LOST_TO_FALSE_POSITIVE,
                       comp_res.MatchChanges.FALSE_POSITIVE_TO_LOST,
                       comp_res.MatchChanges.REMAIN_FALSE_POSITIVE,
                       comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_POSITIVE,
                       comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_NEGATIVE,
                       comp_res.MatchChanges.FALSE_POSITIVE_TO_FALSE_NEGATIVE,
                       comp_res.MatchChanges.LOST_TO_TRUE_NEGATIVE,
                       comp_res.MatchChanges.TRUE_NEGATIVE_TO_LOST,
                       comp_res.MatchChanges.REMAIN_TRUE_NEGATIVE,
                       comp_res.MatchChanges.TRUE_NEGATIVE_TO_TRUE_POSITIVE,
                       comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_POSITIVE,
                       comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_NEGATIVE,
                       comp_res.MatchChanges.LOST_TO_FALSE_NEGATIVE,
                       comp_res.MatchChanges.FALSE_NEGATIVE_TO_LOST,
                       comp_res.MatchChanges.REMAIN_FALSE_NEGATIVE,
                       comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_POSITIVE,
                       comp_res.MatchChanges.FALSE_NEGATIVE_TO_FALSE_POSITIVE,
                       comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_NEGATIVE]
        kwargs = du.defaults(kwargs, {
            'benchmark_comparison_id': np.random.randint(0, 10),
            'benchmark_result': np.random.randint(10, 20),
            'reference_benchmark_result': np.random.randint(20, 30),
            'match_changes': {np.random.uniform(0, 600): match_types[np.random.randint(0, len(match_types))]
                              for _ in range(100)},
            'settings': {}
        })
        return comp_res.MatchingComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: MatchBenchmarkResult
        :param benchmark_result2: MatchBenchmarkResult
        :return:
        """
        if (not isinstance(benchmark_result1, comp_res.MatchingComparisonResult) or
                not isinstance(benchmark_result2, comp_res.MatchingComparisonResult)):
            self.fail('object was not a MatchingComparisonResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.comparison_id, benchmark_result2.comparison_id)
        self.assertEqual(benchmark_result1.benchmark_result, benchmark_result2.benchmark_result)
        self.assertEqual(benchmark_result1.reference_benchmark_result, benchmark_result2.reference_benchmark_result)
        self.assertEqual(benchmark_result1.match_changes, benchmark_result2.match_changes)
        self.assertEqual(benchmark_result1.new_true_positives, benchmark_result2.new_true_positives)
        self.assertEqual(benchmark_result1.new_false_positives, benchmark_result2.new_false_positives)
        self.assertEqual(benchmark_result1.new_true_negatives, benchmark_result2.new_true_negatives)
        self.assertEqual(benchmark_result1.new_false_negatives, benchmark_result2.new_false_negatives)
        self.assertEqual(benchmark_result1.new_missing, benchmark_result2.new_missing)
        self.assertEqual(benchmark_result1.new_found, benchmark_result2.new_found)
        self.assertEqual(benchmark_result1.num_unchanged, benchmark_result2.num_unchanged)
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
            if key is not 'match_changes':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for BSON
        trans_error1 = pickle.loads(s_model1['match_changes'])
        trans_error2 = pickle.loads(s_model2['match_changes'])
        self.assertEqual(set(trans_error1.keys()), set(trans_error2.keys()))
        for key in trans_error1:
            self.assertTrue(np.array_equal(trans_error1[key], trans_error2[key]))

    def test_new_true_positives_is_correct(self):
        subject = self.make_instance()
        new_true_positives = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_POSITIVE or
                    change is comp_res.MatchChanges.TRUE_NEGATIVE_TO_TRUE_POSITIVE or
                    change is comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_POSITIVE or
                    change is comp_res.MatchChanges.LOST_TO_TRUE_POSITIVE):
                new_true_positives += 1
        self.assertEqual(new_true_positives, subject.new_true_positives)

    def test_false_positives_is_correct(self):
        subject = self.make_instance()
        new_false_positives = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_POSITIVE or
                    change is comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_POSITIVE or
                    change is comp_res.MatchChanges.FALSE_NEGATIVE_TO_FALSE_POSITIVE or
                    change is comp_res.MatchChanges.LOST_TO_FALSE_POSITIVE):
                new_false_positives += 1
        self.assertEqual(new_false_positives, subject.new_false_positives)

    def test_true_negatives_is_correct(self):
        subject = self.make_instance()
        new_true_negatives = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.TRUE_POSITIVE_TO_TRUE_NEGATIVE or
                    change is comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_NEGATIVE or
                    change is comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_NEGATIVE or
                    change is comp_res.MatchChanges.LOST_TO_TRUE_NEGATIVE):
                new_true_negatives += 1
        self.assertEqual(new_true_negatives, subject.new_true_negatives)

    def test_false_negatives_is_correct(self):
        subject = self.make_instance()
        new_false_negatives = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_NEGATIVE or
                    change is comp_res.MatchChanges.FALSE_POSITIVE_TO_FALSE_NEGATIVE or
                    change is comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_NEGATIVE or
                    change is comp_res.MatchChanges.LOST_TO_FALSE_NEGATIVE):
                new_false_negatives += 1
        self.assertEqual(new_false_negatives, subject.new_false_negatives)

    def test_new_missing_is_correct(self):
        subject = self.make_instance()
        new_missing = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.TRUE_POSITIVE_TO_LOST or
                    change is comp_res.MatchChanges.FALSE_POSITIVE_TO_LOST or
                    change is comp_res.MatchChanges.TRUE_NEGATIVE_TO_LOST or
                    change is comp_res.MatchChanges.FALSE_NEGATIVE_TO_LOST):
                new_missing += 1
        self.assertEqual(new_missing, subject.new_missing)

    def test_new_found_is_correct(self):
        subject = self.make_instance()
        new_found = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.LOST_TO_TRUE_POSITIVE or
                    change is comp_res.MatchChanges.LOST_TO_FALSE_POSITIVE or
                    change is comp_res.MatchChanges.LOST_TO_TRUE_NEGATIVE or
                    change is comp_res.MatchChanges.LOST_TO_FALSE_NEGATIVE):
                new_found += 1
        self.assertEqual(new_found, subject.new_found)

    def test_unchanged_is_correct(self):
        subject = self.make_instance()
        num_unchanged = 0
        for change in subject.match_changes.values():
            if (change is comp_res.MatchChanges.REMAIN_TRUE_POSITIVE or
                    change is comp_res.MatchChanges.REMAIN_FALSE_POSITIVE or
                    change is comp_res.MatchChanges.REMAIN_TRUE_NEGATIVE or
                    change is comp_res.MatchChanges.REMAIN_FALSE_NEGATIVE):
                num_unchanged += 1
        self.assertEqual(num_unchanged, subject.num_unchanged)
