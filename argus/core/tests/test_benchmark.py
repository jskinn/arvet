# Copyright (c) 2017, John Skinner
import unittest
import argus.util.dict_utils as du
import argus.database.tests.test_entity as entity_test
import argus.core.benchmark


class TestBenchmarkResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return argus.core.benchmark.BenchmarkResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {'benchmark_id': 1, 'trial_result_id': 2, 'success': True})
        return argus.core.benchmark.BenchmarkResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if (not isinstance(benchmark_result1, argus.core.benchmark.BenchmarkResult) or
                not isinstance(benchmark_result2, argus.core.benchmark.BenchmarkResult)):
            self.fail('object was not a BenchmarkResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)


class TestFailedBenchmark(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return argus.core.benchmark.FailedBenchmark

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': 1,
            'trial_result_id': 2,
            'reason': 'For test purposes'
        })
        return argus.core.benchmark.FailedBenchmark(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if (not isinstance(benchmark_result1, argus.core.benchmark.FailedBenchmark) or
                not isinstance(benchmark_result2, argus.core.benchmark.FailedBenchmark)):
            self.fail('object was not a FailedBenchmark')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.reason, benchmark_result2.reason)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)
