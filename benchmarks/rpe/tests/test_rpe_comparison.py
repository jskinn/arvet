#Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import pickle
import util.dict_utils as du
import database.tests.test_entity as entity_test
import core.benchmark_comparison
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
            'difference_in_translational_error': {np.random.uniform(0, 600): np.random.uniform(-100, 100)
                                                  for _ in range(100)},
            'difference_in_rotational_error': {np.random.uniform(0, 600): np.random.uniform(-100, 100)
                                               for _ in range(100)},
            'settings': {
                'offset': np.random.randint(40, 50),
                'max_difference': np.random.randint(50, 60)
            }
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
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.comparison_id, benchmark_result2.comparison_id)
        self.assertEqual(benchmark_result1.benchmark_result, benchmark_result2.benchmark_result)
        self.assertEqual(benchmark_result1.reference_benchmark_result, benchmark_result2.reference_benchmark_result)
        self.assertEqual(benchmark_result1.translational_error_difference,
                         benchmark_result2.translational_error_difference)
        self.assertEqual(benchmark_result1.rotational_error_difference,
                         benchmark_result2.rotational_error_difference)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'trans_error_diff' and key is not 'rot_error_diff':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for BSON
        trans_error1 = pickle.loads(s_model1['trans_error_diff'])
        trans_error2 = pickle.loads(s_model2['trans_error_diff'])
        self.assertEqual(set(trans_error1.keys()), set(trans_error2.keys()))
        for key in trans_error1:
            self.assertTrue(np.array_equal(trans_error1[key], trans_error2[key]))

        rot_error1 = pickle.loads(s_model1['rot_error_diff'])
        rot_error2 = pickle.loads(s_model2['rot_error_diff'])
        self.assertEqual(set(rot_error1.keys()), set(rot_error2.keys()))
        for key in rot_error1:
            self.assertTrue(np.array_equal(rot_error1[key], rot_error2[key]))


class TestRPEBenchmarkComparison(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return rpe_comp.RPEBenchmarkComparison

    def make_instance(self, *args, **kwargs):
        return rpe_comp.RPEBenchmarkComparison(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: RPEBenchmarkComparison
        :param benchmark2: RPEBenchmarkComparison
        :return:
        """
        if (not isinstance(benchmark1, rpe_comp.RPEBenchmarkComparison) or
                not isinstance(benchmark2, rpe_comp.RPEBenchmarkComparison)):
            self.fail('object was not a RPEBenchmarkComparison')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.offset, benchmark2.offset)
        self.assertEqual(benchmark1.max_difference, benchmark2.max_difference)

    def test_comparison_returns_comparison_benchmark_result(self):
        random = np.random.RandomState(6188)
        ref_trans_error = {random.uniform(0, 600): random.uniform(-100, 100) for _ in range(100)}
        ref_rot_error = {time: random.uniform(-np.pi / 2, np.pi / 2) for time in ref_trans_error}
        ref_benchmark_result = rpe_res.BenchmarkRPEResult(benchmark_id=random.randint(0, 10),
                                                          trial_result_id=random.randint(10, 20),
                                                          translational_error=ref_trans_error,
                                                          rotational_error=ref_rot_error,
                                                          rpe_settings={})
        comp_benchmark_result = rpe_res.BenchmarkRPEResult(benchmark_id=random.randint(20, 30),
                                                           trial_result_id=random.randint(30, 40),
                                                           translational_error=ref_trans_error,
                                                           rotational_error=ref_rot_error,
                                                           rpe_settings={})
        comparison_benchmark = rpe_comp.RPEBenchmarkComparison()
        comparison_result = comparison_benchmark.compare_results(comp_benchmark_result, ref_benchmark_result)
        self.assertIsInstance(comparison_result, core.benchmark_comparison.BenchmarkComparisonResult)
        self.assertEqual(comparison_benchmark.identifier, comparison_result.comparison_id)
        self.assertEqual(comp_benchmark_result.identifier, comparison_result.benchmark_result)
        self.assertEqual(ref_benchmark_result.identifier, comparison_result.reference_benchmark_result)

    def test_comparison_returns_error_diff(self):
        random = np.random.RandomState(19687)
        ref_trans_error = {random.uniform(0, 600): random.uniform(-100, 100) for _ in range(100)}
        ref_rot_error = {time: random.uniform(-np.pi/2, np.pi/2) for time in ref_trans_error}

        ref_benchmark_result = rpe_res.BenchmarkRPEResult(benchmark_id=random.randint(0, 10),
                                                          trial_result_id=random.randint(10, 20),
                                                          translational_error=ref_trans_error,
                                                          rotational_error=ref_rot_error,
                                                          rpe_settings={})
        added_trans_error = {}
        added_rot_error = {}
        test_trans_error = {random.uniform(0, 600) + 1000: random.uniform(-100, 100) for _ in range(10)}
        test_rot_error = {random.uniform(0, 600) + 1000: random.uniform(-100, 100) for _ in range(10)}
        for time in ref_trans_error:
            added_trans_error[time] = random.uniform(-50, 50)
            added_rot_error[time] = random.uniform(-np.pi/6, np.pi/6)
            noisy_time = time + random.uniform(-0.005, 0.005)
            test_trans_error[noisy_time] = ref_trans_error[time] + added_trans_error[time]
            test_rot_error[noisy_time] = ref_rot_error[time] + added_rot_error[time]
        subject_benchmark_result = rpe_res.BenchmarkRPEResult(benchmark_id=random.randint(20, 30),
                                                              trial_result_id=random.randint(30, 40),
                                                              translational_error=test_trans_error,
                                                              rotational_error=test_rot_error,
                                                              rpe_settings={})
        comparison_benchmark = rpe_comp.RPEBenchmarkComparison()
        comparison_result = comparison_benchmark.compare_results(subject_benchmark_result, ref_benchmark_result)

        for time, error in added_trans_error.items():
            self.assertIn(time, comparison_result.translational_error_difference)
            self.assertAlmostEqual(-1 * error, comparison_result.translational_error_difference[time])
        for time, error in added_rot_error.items():
            self.assertIn(time, comparison_result.translational_error_difference)
            self.assertAlmostEqual(-1 * error, comparison_result.rotational_error_difference[time])
