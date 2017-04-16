import numpy as np
import unittest
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.ate.ate_result as ate_res
import benchmarks.ate.absolute_trajectory_error_comparison as ate_comp


class TestATEBenchmarkComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return ate_comp.ATEBenchmarkComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_comparison_id': np.random.randint(0, 10),
            'benchmark_result': np.random.randint(10, 20),
            'reference_benchmark_result': np.random.randint(20, 30),
            'difference_in_translational_error': np.random.rand(100, 10)
        })
        return ate_comp.ATEBenchmarkComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: ATEBenchmarkComparisonResult
        :param benchmark_result2: ATEBenchmarkComparisonResult
        :return:
        """
        if (not isinstance(benchmark_result1, ate_comp.ATEBenchmarkComparisonResult) or
                not isinstance(benchmark_result2, ate_comp.ATEBenchmarkComparisonResult)):
            self.fail('object was not a BenchmarkATEResult')
        self.assertEquals(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEquals(benchmark_result1.success, benchmark_result2.success)
        self.assertEquals(benchmark_result1.comparison_id, benchmark_result2.comparison_id)
        self.assertEquals(benchmark_result1.benchmark_result, benchmark_result2.benchmark_result)
        self.assertEquals(benchmark_result1.reference_benchmark_result, benchmark_result2.reference_benchmark_result)
        self.assertTrue(np.array_equal(benchmark_result1.translational_error_difference,
                                       benchmark_result2.translational_error_difference),
                        "Translational error differences were not equal")


class TestATEBenchmarkComparison(unittest.TestCase):

    def test_comparison_returns_error_diff(self):
        benchmark_result1 = ate_res.BenchmarkATEResult(benchmark_id=np.random.randint(0, 10),
                                                       trial_result_id=np.random.randint(10, 20),
                                                       translational_error=np.random.rand(100, 10),
                                                       ate_settings={})
        benchmark_result2 = ate_res.BenchmarkATEResult(benchmark_id=np.random.randint(20, 30),
                                                       trial_result_id=np.random.randint(30, 40),
                                                       translational_error=np.random.rand(100, 10),
                                                       ate_settings={})
        comparison_benchmark = ate_comp.ATEBenchmarkComparison()
        comparison_result = comparison_benchmark.compare_trial_results(benchmark_result1, benchmark_result2)
        self.assertEquals(comparison_benchmark.identifier, comparison_result.comparison_id)
        self.assertEquals(benchmark_result1.identifier, comparison_result.benchmark_result)
        self.assertEquals(benchmark_result2.identifier, comparison_result.reference_benchmark_result)
        self.assertTrue(np.array_equal(comparison_result.translational_error_difference,
                                       benchmark_result2.translational_error - benchmark_result1.translational_error),
                        "Error difference was not equal to the change in error")
