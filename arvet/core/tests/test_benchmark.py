# Copyright (c) 2017, John Skinner
import unittest
import arvet.util.dict_utils as du
import arvet.database.tests.test_entity as entity_test
import arvet.core.benchmark


class TestBenchmarkResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return arvet.core.benchmark.BenchmarkResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {'benchmark_id': 1, 'trial_result_ids': [2, 3, 4], 'success': True})
        return arvet.core.benchmark.BenchmarkResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if (not isinstance(benchmark_result1, arvet.core.benchmark.BenchmarkResult) or
                not isinstance(benchmark_result2, arvet.core.benchmark.BenchmarkResult)):
            self.fail('object was not a BenchmarkResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_results, benchmark_result2.trial_results)


class TestFailedBenchmark(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return arvet.core.benchmark.FailedBenchmark

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': 1,
            'trial_result_ids': [2, 3, 4],
            'reason': 'For test purposes'
        })
        return arvet.core.benchmark.FailedBenchmark(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if (not isinstance(benchmark_result1, arvet.core.benchmark.FailedBenchmark) or
                not isinstance(benchmark_result2, arvet.core.benchmark.FailedBenchmark)):
            self.fail('object was not a FailedBenchmark')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.reason, benchmark_result2.reason)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_results, benchmark_result2.trial_results)
