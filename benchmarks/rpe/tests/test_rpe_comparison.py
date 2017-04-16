import numpy as np
import unittest
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.rpe.rpe_result as rpe_res
import benchmarks.rpe.relative_pose_error_comparison as rpe_comp


class TestRPEBenchmarkComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return rpe_comp.RPEBenchmarkComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_comparison_id': np.random.randint(0, 10),
            'benchmark_result': np.random.randint(10, 20),
            'reference_benchmark_result': np.random.randint(20, 30),
            'difference_in_translational_error': np.random.rand(100, 10),
            'difference_in_rotational_error': np.random.rand(100, 10)
        })
        return rpe_comp.RPEBenchmarkComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: RPEBenchmarkComparisonResult
        :param benchmark_result2: RPEBenchmarkComparisonResult
        :return:
        """
        if (not isinstance(benchmark_result1, rpe_comp.RPEBenchmarkComparisonResult) or
                not isinstance(benchmark_result2, rpe_comp.RPEBenchmarkComparisonResult)):
            self.fail('object was not a BenchmarkRPEResult')
        self.assertEquals(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEquals(benchmark_result1.success, benchmark_result2.success)
        self.assertEquals(benchmark_result1.comparison_id, benchmark_result2.comparison_id)
        self.assertEquals(benchmark_result1.benchmark_result, benchmark_result2.benchmark_result)
        self.assertEquals(benchmark_result1.reference_benchmark_result, benchmark_result2.reference_benchmark_result)
        self.assertTrue(np.array_equal(benchmark_result1.translational_error_difference,
                                       benchmark_result2.translational_error_difference),
                        "Translational error differences were not equal")
        self.assertTrue(np.array_equal(benchmark_result1.rotational_error_difference,
                                       benchmark_result2.rotational_error_difference),
                        "Rotational error differences were not equal")


class TestRPEBenchmarkComparison(unittest.TestCase):

    def test_comparison_returns_error_diff(self):
        benchmark_result1 = rpe_res.BenchmarkRPEResult(benchmark_id=np.random.randint(0, 10),
                                                       trial_result_id=np.random.randint(10, 20),
                                                       translational_error=np.random.rand(100, 10),
                                                       rotational_error=np.random.rand(100, 10),
                                                       rpe_settings={})
        benchmark_result2 = rpe_res.BenchmarkRPEResult(benchmark_id=np.random.randint(20, 30),
                                                       trial_result_id=np.random.randint(30, 40),
                                                       translational_error=np.random.rand(100, 10),
                                                       rotational_error=np.random.rand(100, 10),
                                                       rpe_settings={})
        comparison_benchmark = rpe_comp.RPEBenchmarkComparison()
        comparison_result = comparison_benchmark.compare_trial_results(benchmark_result1, benchmark_result2)
        self.assertEquals(comparison_benchmark.identifier, comparison_result.comparison_id)
        self.assertEquals(benchmark_result1.identifier, comparison_result.benchmark_result)
        self.assertEquals(benchmark_result2.identifier, comparison_result.reference_benchmark_result)
        self.assertTrue(np.array_equal(comparison_result.translational_error_difference,
                                       benchmark_result2.translational_error - benchmark_result1.translational_error),
                        "Translational error difference was not equal to the change in error")
        self.assertTrue(np.array_equal(comparison_result.rotational_error_difference,
                                       benchmark_result2.rotational_error - benchmark_result1.rotational_error),
                        "Rotational error difference was not equal to the change in error")
