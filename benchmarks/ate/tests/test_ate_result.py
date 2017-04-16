import numpy as np
import unittest
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.ate.ate_result as ate_res


class TestBenchmarkATEResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return ate_res.BenchmarkATEResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_id': np.random.randint(10, 20),
            'translational_error': np.random.rand(100, 10),
            'ate_settings': {}
        })
        return ate_res.BenchmarkATEResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: BenchmarkATEResult
        :param benchmark_result2: BenchmarkATEResult
        :return:
        """
        if (not isinstance(benchmark_result1, ate_res.BenchmarkATEResult) or
                not isinstance(benchmark_result2, ate_res.BenchmarkATEResult)):
            self.fail('object was not a BenchmarkATEResult')
        self.assertEquals(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEquals(benchmark_result1.success, benchmark_result2.success)
        self.assertEquals(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEquals(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertEquals(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertTrue(np.array_equal(benchmark_result1.translational_error, benchmark_result2.translational_error),
                        "Translational errors were not equal")
