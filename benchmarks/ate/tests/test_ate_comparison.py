# Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import pickle
import util.dict_utils as du
import database.tests.test_entity as entity_test
import core.benchmark_comparison
import benchmarks.ate.ate_result as ate_res
import benchmarks.ate.absolute_trajectory_error_comparison as ate_comp


class TestATEBenchmarkComparison(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return ate_comp.ATEBenchmarkComparison

    def make_instance(self, *args, **kwargs):
        return ate_comp.ATEBenchmarkComparison(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: ATEBenchmarkComparison
        :param benchmark2: ATEBenchmarkComparison
        :return:
        """
        if (not isinstance(benchmark1, ate_comp.ATEBenchmarkComparison) or
                not isinstance(benchmark2, ate_comp.ATEBenchmarkComparison)):
            self.fail('object was not a BenchmarkATE')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.offset, benchmark2.offset)
        self.assertEqual(benchmark1.max_difference, benchmark2.max_difference)

    def test_comparison_returns_comparison_result(self):
        random = np.random.RandomState(1687)
        benchmark_result1 = ate_res.BenchmarkATEResult(benchmark_id=random.randint(0, 10),
                                                       trial_result_id=random.randint(10, 20),
                                                       translational_error={random.uniform(0, 600):
                                                                            random.uniform(-100, 100)
                                                                            for _ in range(100)},
                                                       ate_settings={})
        benchmark_result2 = ate_res.BenchmarkATEResult(benchmark_id=random.randint(20, 30),
                                                       trial_result_id=random.randint(30, 40),
                                                       translational_error={random.uniform(0, 600):
                                                                            random.uniform(-100, 100)
                                                                            for _ in range(100)},
                                                       ate_settings={})

        comparison_benchmark = ate_comp.ATEBenchmarkComparison()
        comparison_result = comparison_benchmark.compare_results(benchmark_result1, benchmark_result2)
        self.assertIsInstance(comparison_result, core.benchmark_comparison.BenchmarkComparisonResult)
        self.assertEqual(comparison_benchmark.identifier, comparison_result.comparison_id)
        self.assertEqual(benchmark_result1.identifier, comparison_result.benchmark_result)
        self.assertEqual(benchmark_result2.identifier, comparison_result.reference_benchmark_result)

    def test_comparison_returns_error_diff(self):
        random = np.random.RandomState(1425)

        ref_error = {random.uniform(0, 600): random.uniform(-100, 100) for _ in range(100)}
        ref_benchmark_result = ate_res.BenchmarkATEResult(benchmark_id=random.randint(0, 10),
                                                          trial_result_id=random.randint(10, 20),
                                                          translational_error=ref_error,
                                                          ate_settings={})

        # Add error to the tested benchmark result
        added_error = {}
        test_error = {random.uniform(0, 600) + 1000: random.uniform(-100, 100) for _ in range(10)}
        for time, error in ref_error.items():
            added_error[time] = random.uniform(-50, 50)
            test_error[time + random.uniform(-0.005, 0.005)] = error + added_error[time]
        subject_benchmark_result = ate_res.BenchmarkATEResult(benchmark_id=random.randint(20, 30),
                                                              trial_result_id=random.randint(30, 40),
                                                              translational_error=test_error,
                                                              ate_settings={})

        comparison_benchmark = ate_comp.ATEBenchmarkComparison()
        comparison_result = comparison_benchmark.compare_results(subject_benchmark_result, ref_benchmark_result)

        for time, error in added_error.items():
            self.assertIn(time, comparison_result.translational_error_difference)
            self.assertAlmostEqual(-1 * error, comparison_result.translational_error_difference[time])


class TestATEBenchmarkComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return ate_comp.ATEBenchmarkComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_comparison_id': np.random.randint(0, 10),
            'benchmark_result': np.random.randint(10, 20),
            'reference_benchmark_result': np.random.randint(20, 30),
            'difference_in_translational_error': {np.random.uniform(0, 600): np.random.uniform(-1000, 1000)
                                                  for _ in range(100)},
            'settings': {
                'offset': np.random.randint(40, 50),
                'max_difference': np.random.randint(50, 60)
            }
        })
        return ate_comp.ATEBenchmarkComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark1: ATEBenchmarkComparisonResult
        :param benchmark2: ATEBenchmarkComparisonResult
        :return:
        """
        if (not isinstance(benchmark1, ate_comp.ATEBenchmarkComparisonResult) or
                not isinstance(benchmark2, ate_comp.ATEBenchmarkComparisonResult)):
            self.fail('object was not a BenchmarkATEResult')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.success, benchmark2.success)
        self.assertEqual(benchmark1.comparison_id, benchmark2.comparison_id)
        self.assertEqual(benchmark1.benchmark_result, benchmark2.benchmark_result)
        self.assertEqual(benchmark1.reference_benchmark_result, benchmark2.reference_benchmark_result)
        self.assertEqual(benchmark1.translational_error_difference,
                         benchmark2.translational_error_difference)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'trans_error_diff':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for BSON
        trans_error1 = pickle.loads(s_model1['trans_error_diff'])
        trans_error2 = pickle.loads(s_model2['trans_error_diff'])
        self.assertEqual(set(trans_error1.keys()), set(trans_error2.keys()))
        for key in trans_error1:
            self.assertTrue(np.array_equal(trans_error1[key], trans_error2[key]))
