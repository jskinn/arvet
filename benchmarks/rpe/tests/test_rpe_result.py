# Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import pickle
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.rpe.rpe_result as rpe_res


class TestBenchmarkRPEResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return rpe_res.BenchmarkRPEResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_id': np.random.randint(10, 20),
            'translational_error': {np.random.uniform(0, 600): np.random.uniform(-100, 100) for _ in range(100)},
            'rotational_error': {np.random.uniform(0, 600):  np.random.uniform(-np.pi, np.pi) for _ in range(100)},
            'rpe_settings': {}
        })
        return rpe_res.BenchmarkRPEResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: BenchmarkRPEResult
        :param benchmark_result2: BenchmarkRPEResult
        :return:
        """
        if (not isinstance(benchmark_result1, rpe_res.BenchmarkRPEResult) or
                not isinstance(benchmark_result2, rpe_res.BenchmarkRPEResult)):
            self.fail('object was not a BenchmarkRPEResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertEqual(benchmark_result1.translational_error, benchmark_result2.translational_error)
        self.assertEqual(benchmark_result1.rotational_error, benchmark_result2.rotational_error)
        self.assertEqual(benchmark_result1.settings, benchmark_result2.settings)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'trans_error' and key is not 'rot_error':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for BSON
        trans_error1 = pickle.loads(s_model1['trans_error'])
        trans_error2 = pickle.loads(s_model2['trans_error'])
        self.assertEqual(set(trans_error1.keys()), set(trans_error2.keys()))
        for key in trans_error1:
            self.assertTrue(np.array_equal(trans_error1[key], trans_error2[key]))

        rot_error1 = pickle.loads(s_model1['rot_error'])
        rot_error2 = pickle.loads(s_model2['rot_error'])
        self.assertEqual(set(rot_error1.keys()), set(rot_error2.keys()))
        for key in rot_error1:
            self.assertTrue(np.array_equal(rot_error1[key], rot_error2[key]))

    def test_trans_mean_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.translational_error.values()))
        self.assertEqual(np.mean(trans_error), subject.trans_mean)

    def test_trans_median_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.translational_error.values()))
        self.assertEqual(np.median(trans_error), subject.trans_median)

    def test_trans_std_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.translational_error.values()))
        self.assertEqual(np.std(trans_error), subject.trans_std)

    def test_trans_min_is_correct(self):
        subject = self.make_instance()
        min_ = None
        for error in subject.translational_error.values():
            if min_ is None or error < min_:
                min_ = error
        self.assertEqual(min_, subject.trans_min)

    def test_trans_max_is_correct(self):
        subject = self.make_instance()
        max_ = None
        for error in subject.translational_error.values():
            if max_ is None or error > max_:
                max_ = error
        self.assertEqual(max_, subject.trans_max)

    def test_rot_mean_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.rotational_error.values()))
        self.assertEqual(np.mean(trans_error), subject.rot_mean)

    def test_rot_median_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.rotational_error.values()))
        self.assertEqual(np.median(trans_error), subject.rot_median)

    def test_rot_std_is_correct(self):
        subject = self.make_instance()
        trans_error = np.array(list(subject.rotational_error.values()))
        self.assertEqual(np.std(trans_error), subject.rot_std)

    def test_rot_min_is_correct(self):
        subject = self.make_instance()
        min_ = None
        for error in subject.rotational_error.values():
            if min_ is None or error < min_:
                min_ = error
        self.assertEqual(min_, subject.rot_min)

    def test_rot_max_is_correct(self):
        subject = self.make_instance()
        max_ = None
        for error in subject.rotational_error.values():
            if max_ is None or error > max_:
                max_ = error
        self.assertEqual(max_, subject.rot_max)
