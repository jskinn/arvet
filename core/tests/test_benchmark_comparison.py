#Copyright (c) 2017, John Skinner
import unittest
import util.dict_utils as du
import database.tests.test_entity as entity_test
import core.benchmark_comparison


class TestBenchmarkComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return core.benchmark_comparison.BenchmarkComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_comparison_id': 1,
            'benchmark_result': 2,
            'reference_benchmark_result': 3,
            'success': True
        })
        return core.benchmark_comparison.BenchmarkComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if (not isinstance(benchmark_result1, core.benchmark_comparison.BenchmarkComparisonResult) or
                not isinstance(benchmark_result2, core.benchmark_comparison.BenchmarkComparisonResult)):
            self.fail('object was not a BenchmarkResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.comparison_id, benchmark_result2.comparison_id)
        self.assertEqual(benchmark_result1.benchmark_result, benchmark_result2.benchmark_result)
        self.assertEqual(benchmark_result1.reference_benchmark_result, benchmark_result2.reference_benchmark_result)
