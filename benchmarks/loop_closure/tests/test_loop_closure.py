import numpy as np
import unittest
import util.transform as tf
import core.benchmark
import benchmarks.matching.matching_result as match_res
import benchmarks.loop_closure.loop_closure as lc


class MockTrialResult:

    def __init__(self, gt_trajectory, loop_closures):
        self._gt_traj = gt_trajectory
        self._closures = loop_closures

    @property
    def identifier(self):
        return 'ThisIsAMockTrialResult'

    @property
    def ground_truth_trajectory(self):
        return self._gt_traj

    @ground_truth_trajectory.setter
    def ground_truth_trajectory(self, ground_truth_trajectory):
        self._gt_traj = ground_truth_trajectory

    @property
    def loop_closures(self):
        return self._closures

    @loop_closures.setter
    def loop_closures(self, loop_closures):
        self._closures = loop_closures

    def get_ground_truth_camera_poses(self):
        return self.ground_truth_trajectory

    def get_loop_closures(self):
        return self.loop_closures


class TestBenchmarkLoopClosure(unittest.TestCase):

    def test_benchmark_results_returns_a_benchmark_result(self):
        trial_result = MockTrialResult(gt_trajectory={
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1))
        }, loop_closures={})

        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)
        self.assertIsInstance(result, core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertIsInstance(result, match_res.MatchBenchmarkResult)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(trial_result.identifier, result.trial_result)

    def test_benchmark_measures_match_for_all_stamps(self):
        random = np.random.RandomState(1563)
        trajectory = {random.uniform(0, 600): tf.Transform(location=(100, 100, 0),
                                                           rotation=(0, 0, 0, 1))
                      for _ in range(100)}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures={})
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)

        for stamp in trajectory.keys():
            self.assertIn(stamp, result.matches)

    def test_benchmark_detects_true_positives(self):
        # Create a trial result with a correct loop closure
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1))
        }
        closures = {10.66667: 1.33333}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures=closures)

        # Perform the benchmark
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.TRUE_POSITIVE, result.matches[10.66667])

    def test_benchmark_detects_false_positives(self):
        # Create a trial result with an incorrect loop closure
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(-100, -100, 0), rotation=(0, 0, 0, 1))
        }
        closures = {10.66667: 1.33333}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures=closures)

        # Perform the benchmark
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.FALSE_POSITIVE, result.matches[10.66667])

    def test_benchmark_detects_true_negative(self):
        # Create a trial result without a correct loop closure
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(-100, -100, 0), rotation=(0, 0, 0, 1))
        }
        closures = {}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures=closures)

        # Perform the benchmark
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.TRUE_NEGATIVE, result.matches[10.66667])
        self.assertEqual(match_res.MatchType.TRUE_NEGATIVE, result.matches[1.33333])

    def test_benchmark_detects_false_negative(self):
        # Create a trial result with a correct loop closure
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1))
        }
        closures = {}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures=closures)

        # Perform the benchmark
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.FALSE_NEGATIVE, result.matches[10.66667])

    def test_distance_threshold_determines_acceptable_matches(self):
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(110, 100, 0), rotation=(0, 0, 0, 1)),
            15.33333: tf.Transform(location=(-100, 100, 0), rotation=(0, 0, 0, 1)),
            20.66667: tf.Transform(location=(-110, 100, 0), rotation=(0, 0, 0, 1))
        }
        closures = {10.66667: 1.33333}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures=closures)

        # Perform the benchmark with a larger threshold
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.TRUE_POSITIVE, result.matches[10.66667])
        self.assertEqual(match_res.MatchType.FALSE_NEGATIVE, result.matches[20.66667])

        # Try again with a smaller threshold, should become a false positive
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=1)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.FALSE_POSITIVE, result.matches[10.66667])
        self.assertEqual(match_res.MatchType.TRUE_NEGATIVE, result.matches[20.66667])

    def test_matches_too_close_are_trivial_matches(self):
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            3.66667: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            14.33333: tf.Transform(location=(-100, 100, 0), rotation=(0, 0, 0, 1)),
            15.66667: tf.Transform(location=(-100, 100, 0), rotation=(0, 0, 0, 1))
        }
        closures = {3.66667: 1.33333}
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures=closures)

        # Perform the benchmark with a larger threshold
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20, trivial_closure_index_distance=10)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.FALSE_POSITIVE, result.matches[3.66667])
        self.assertEqual(match_res.MatchType.TRUE_NEGATIVE, result.matches[15.66667])

        # Try again with a smaller threshold, should become a true positive,
        # since the indexes are further appart than the threshold
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20, trivial_closure_index_distance=1)
        result = benchmark.benchmark_results(trial_result)
        self.assertEqual(match_res.MatchType.TRUE_POSITIVE, result.matches[3.66667])
        self.assertEqual(match_res.MatchType.FALSE_NEGATIVE, result.matches[15.66667])

    def test_benchmark_accepts_any_of_multiple_closures(self):
        # Create a trial result with multiple valid closures
        trajectory = {
            1.33333: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            5.33333: tf.Transform(location=(-100, -100, 0), rotation=(0, 0, 0, 1)),
            10.66667: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            12.33333: tf.Transform(location=(-100, -100, 0), rotation=(0, 0, 0, 1)),
            15: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1)),
            17.66667: tf.Transform(location=(-100, -100, 0), rotation=(0, 0, 0, 1)),
            20.66667: tf.Transform(location=(100, 100, 0), rotation=(0, 0, 0, 1))
        }
        trial_result = MockTrialResult(gt_trajectory=trajectory, loop_closures={})
        benchmark = lc.BenchmarkLoopClosure(distance_threshold=20)

        # Simple cartesian product to get all the different combinations of closures for the location
        group1 = [20.66667, 15, 10.66667, 1.33333]
        valid_pairs = [(idx, closure) for idx in group1 for closure in group1 if closure < idx]

        # Test different possible closures
        for idx, closure in valid_pairs:
            trial_result.loop_closures = {idx: closure}
            result = benchmark.benchmark_results(trial_result)
            self.assertEqual(match_res.MatchType.TRUE_POSITIVE, result.matches[idx])
