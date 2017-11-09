# Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import database.tests.test_entity
import core.benchmark_comparison
import benchmarks.matching.matching_result as match_res
import benchmarks.matching.match_comparison as match_comp
import benchmarks.matching.match_comparison_result as match_comp_res


def create_matches(random_state, duration=600, length=100):
    match_types = [match_res.MatchType.TRUE_POSITIVE,
                   match_res.MatchType.FALSE_POSITIVE,
                   match_res.MatchType.TRUE_NEGATIVE,
                   match_res.MatchType.FALSE_NEGATIVE]
    return {random_state.uniform(0, duration):
            match_types[random_state.randint(0, len(match_types))]
            for _ in range(length)}


class TestMatchComparison(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return match_comp.BenchmarkMatchingComparison

    def make_instance(self, *args, **kwargs):
        return match_comp.BenchmarkMatchingComparison(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: BenchmarkMatchingComparison
        :param benchmark2: BenchmarkMatchingComparison
        :return:
        """
        if (not isinstance(benchmark1, match_comp.BenchmarkMatchingComparison) or
                not isinstance(benchmark2, match_comp.BenchmarkMatchingComparison)):
            self.fail('object was not a BenchmarkMatchingComparison')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.offset, benchmark2.offset)
        self.assertEqual(benchmark1.max_difference, benchmark2.max_difference)

    def test_comparison_returns_comparison_result(self):
        random = np.random.RandomState(1687)
        ref_benchmark_result = match_res.MatchBenchmarkResult(benchmark_id=random.randint(0, 10),
                                                             trial_result_id=random.randint(10, 20),
                                                             matches=create_matches(random),
                                                             settings={})
        test_benchmark_result = match_res.MatchBenchmarkResult(benchmark_id=random.randint(0, 10),
                                                              trial_result_id=random.randint(10, 20),
                                                              matches=create_matches(random),
                                                              settings={})
        comparison_benchmark = match_comp.BenchmarkMatchingComparison()
        comparison_result = comparison_benchmark.compare_results(test_benchmark_result, ref_benchmark_result)
        self.assertIsInstance(comparison_result, core.benchmark_comparison.BenchmarkComparisonResult)
        self.assertEqual(comparison_benchmark.identifier, comparison_result.comparison_id)
        self.assertEqual(test_benchmark_result.identifier, comparison_result.benchmark_result)
        self.assertEqual(ref_benchmark_result.identifier, comparison_result.reference_benchmark_result)

    def test_comparison_returns_change(self):
        random = np.random.RandomState(1425)
        self.maxDiff = None
        ref_matches = {
            1: match_res.MatchType.TRUE_POSITIVE,
            2: match_res.MatchType.TRUE_POSITIVE,
            3: match_res.MatchType.TRUE_POSITIVE,
            4: match_res.MatchType.TRUE_POSITIVE,
            5: match_res.MatchType.TRUE_POSITIVE,
            #6: Lost,
            7: match_res.MatchType.FALSE_POSITIVE,
            8: match_res.MatchType.FALSE_POSITIVE,
            9: match_res.MatchType.FALSE_POSITIVE,
            10: match_res.MatchType.FALSE_POSITIVE,
            11: match_res.MatchType.FALSE_POSITIVE,
            #12: Lost,
            13: match_res.MatchType.TRUE_NEGATIVE,
            14: match_res.MatchType.TRUE_NEGATIVE,
            15: match_res.MatchType.TRUE_NEGATIVE,
            16: match_res.MatchType.TRUE_NEGATIVE,
            17: match_res.MatchType.TRUE_NEGATIVE,
            #18: Lost,
            19: match_res.MatchType.FALSE_NEGATIVE,
            20: match_res.MatchType.FALSE_NEGATIVE,
            21: match_res.MatchType.FALSE_NEGATIVE,
            22: match_res.MatchType.FALSE_NEGATIVE,
            23: match_res.MatchType.FALSE_NEGATIVE,
            #24: Lost

        }
        comp_matches = {
            1: match_res.MatchType.TRUE_POSITIVE,
            2: match_res.MatchType.FALSE_POSITIVE,
            3: match_res.MatchType.TRUE_NEGATIVE,
            4: match_res.MatchType.FALSE_NEGATIVE,
            #5: Lost,
            6: match_res.MatchType.TRUE_POSITIVE,
            7: match_res.MatchType.TRUE_POSITIVE,
            8: match_res.MatchType.FALSE_POSITIVE,
            9: match_res.MatchType.TRUE_NEGATIVE,
            10: match_res.MatchType.FALSE_NEGATIVE,
            #11: Lost,
            12: match_res.MatchType.FALSE_POSITIVE,
            13: match_res.MatchType.TRUE_POSITIVE,
            14: match_res.MatchType.FALSE_POSITIVE,
            15: match_res.MatchType.TRUE_NEGATIVE,
            16: match_res.MatchType.FALSE_NEGATIVE,
            #17: Lost,
            18: match_res.MatchType.TRUE_NEGATIVE,
            19: match_res.MatchType.TRUE_POSITIVE,
            20: match_res.MatchType.FALSE_POSITIVE,
            21: match_res.MatchType.TRUE_NEGATIVE,
            22: match_res.MatchType.FALSE_NEGATIVE,
            #23: Lost,
            24: match_res.MatchType.FALSE_NEGATIVE
        }
        expected_changes = {
            1: match_comp_res.MatchChanges.REMAIN_TRUE_POSITIVE,
            2: match_comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_POSITIVE,
            3: match_comp_res.MatchChanges.TRUE_POSITIVE_TO_TRUE_NEGATIVE,
            4: match_comp_res.MatchChanges.TRUE_POSITIVE_TO_FALSE_NEGATIVE,
            5: match_comp_res.MatchChanges.TRUE_POSITIVE_TO_LOST,
            6: match_comp_res.MatchChanges.LOST_TO_TRUE_POSITIVE,
            7: match_comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_POSITIVE,
            8: match_comp_res.MatchChanges.REMAIN_FALSE_POSITIVE,
            9: match_comp_res.MatchChanges.FALSE_POSITIVE_TO_TRUE_NEGATIVE,
            10: match_comp_res.MatchChanges.FALSE_POSITIVE_TO_FALSE_NEGATIVE,
            11: match_comp_res.MatchChanges.FALSE_POSITIVE_TO_LOST,
            12: match_comp_res.MatchChanges.LOST_TO_FALSE_POSITIVE,
            13: match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_TRUE_POSITIVE,
            14: match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_POSITIVE,
            15: match_comp_res.MatchChanges.REMAIN_TRUE_NEGATIVE,
            16: match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_FALSE_NEGATIVE,
            17: match_comp_res.MatchChanges.TRUE_NEGATIVE_TO_LOST,
            18: match_comp_res.MatchChanges.LOST_TO_TRUE_NEGATIVE,
            19: match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_POSITIVE,
            20: match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_FALSE_POSITIVE,
            21: match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_TRUE_NEGATIVE,
            22: match_comp_res.MatchChanges.REMAIN_FALSE_NEGATIVE,
            23: match_comp_res.MatchChanges.FALSE_NEGATIVE_TO_LOST,
            24: match_comp_res.MatchChanges.LOST_TO_FALSE_NEGATIVE
        }

        ref_benchmark_result = match_res.MatchBenchmarkResult(benchmark_id=random.randint(0, 10),
                                                              trial_result_id=random.randint(10, 20),
                                                              matches=ref_matches,
                                                              settings={})
        comp_benchmark_result = match_res.MatchBenchmarkResult(benchmark_id=random.randint(0, 10),
                                                               trial_result_id=random.randint(10, 20),
                                                               matches=comp_matches,
                                                               settings={})
        comparison_benchmark = match_comp.BenchmarkMatchingComparison()
        comparison_result = comparison_benchmark.compare_results(comp_benchmark_result, ref_benchmark_result)
        self.assertEqual(expected_changes, comparison_result.match_changes)
