import numpy as np
import unittest
import pickle
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
            'translational_error': {np.random.rand()*600:  200 * np.random.rand() - 100 for _ in range(100)},
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

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEquals(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'trans_error':
                self.assertEquals(s_model1[key], s_model2[key])

        # Special case for BSON
        trans_error1 = pickle.loads(s_model1['trans_error'])
        trans_error2 = pickle.loads(s_model2['trans_error'])
        self.assertEquals(set(trans_error1.keys()), set(trans_error2.keys()))
        for key in trans_error1:
            self.assertTrue(np.array_equal(trans_error1[key], trans_error2[key]))


    def test_mean_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.translational_error.values()))
        self.assertEquals(np.mean(trans_error), subject.mean)

    def test_min_is_correct(self):
        subject = self.make_instance()
        min_ = None
        for error in subject.translational_error.values():
            if min_ is None or error < min_:
                min_ = error
        self.assertEquals(min_, subject.min)

    def test_max_is_correct(self):
        subject = self.make_instance()
        max_ = None
        for error in subject.translational_error.values():
            if max_ is None or error > max_:
                max_ = error
        self.assertEquals(max_, subject.max)
