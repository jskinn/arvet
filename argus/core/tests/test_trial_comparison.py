# Copyright (c) 2017, John Skinner
import unittest

import argus.database.tests.test_entity as entity_test
import argus.util.dict_utils as du

import argus.core.trial_comparison


class TestTrialComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return argus.core.trial_comparison.TrialComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': 1,
            'trial_result_id': 2,
            'reference_id': 3,
            'success': True
        })
        return argus.core.trial_comparison.TrialComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if (not isinstance(benchmark_result1, argus.core.trial_comparison.TrialComparisonResult) or
                not isinstance(benchmark_result2, argus.core.trial_comparison.TrialComparisonResult)):
            self.fail('object was not a BenchmarkResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertEqual(benchmark_result1.reference_trial_result, benchmark_result2.reference_trial_result)
