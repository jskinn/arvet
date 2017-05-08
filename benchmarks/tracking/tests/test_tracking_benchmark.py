import unittest
import database.tests.test_entity
import core.benchmark
import util.transform as tf
import trials.slam.tracking_state
import benchmarks.tracking.tracking_benchmark as tracking


class MockTrialResult:

    def __init__(self, gt_trajectory, tracking_states):
        self._gt_traj = gt_trajectory
        self._tracking_states = tracking_states

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
    def tracking_states(self):
        return self._tracking_states

    @tracking_states.setter
    def tracking_states(self, tracking_states):
        self._tracking_states = tracking_states

    def get_ground_truth_camera_poses(self):
        return self.ground_truth_trajectory

    def get_tracking_states(self):
        return self.tracking_states


class TestTrackingBenchmark(database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.trajectory = {
            1.3333: tf.Transform(location=(0, 0, 0), rotation=(0, 0, 0, 1)),
            1.6667: tf.Transform(location=(10, 0, 0), rotation=(0, 0, 0, 1)),
            2: tf.Transform(location=(20, 0, 0), rotation=(0, 0, 0, 1)),
            2.3333: tf.Transform(location=(30, 0, 0), rotation=(0, 0, 0, 1)),
            2.6667: tf.Transform(location=(40, 0, 0), rotation=(0, 0, 0, 1)),
            3: tf.Transform(location=(50, 0, 0), rotation=(0, 0, 0, 1)),
            3.3333: tf.Transform(location=(60, 0, 0), rotation=(0, 0, 0, 1)),
            3.6667: tf.Transform(location=(70, 0, 0), rotation=(0, 0, 0, 1)),
            4: tf.Transform(location=(80, 0, 0), rotation=(0, 0, 0, 1)),
            4.3333: tf.Transform(location=(90, 0, 0), rotation=(0, 0, 0, 1)),
            4.6667: tf.Transform(location=(100, 0, 0), rotation=(0, 0, 0, 1))
        }
        self.tracking_states = {
            1.3333: trials.slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: trials.slam.tracking_state.TrackingState.OK,
            2: trials.slam.tracking_state.TrackingState.OK,
            2.3333: trials.slam.tracking_state.TrackingState.LOST,
            2.6667: trials.slam.tracking_state.TrackingState.LOST,
            3: trials.slam.tracking_state.TrackingState.OK,
            3.3333: trials.slam.tracking_state.TrackingState.LOST,
            3.6667: trials.slam.tracking_state.TrackingState.LOST,
            4: trials.slam.tracking_state.TrackingState.LOST,
            4.3333: trials.slam.tracking_state.TrackingState.OK,
            4.6667: trials.slam.tracking_state.TrackingState.LOST
        }
        self.trial_result = MockTrialResult(gt_trajectory=self.trajectory, tracking_states=self.tracking_states)

    def get_class(self):
        return tracking.TrackingBenchmark

    def make_instance(self, *args, **kwargs):
        return tracking.TrackingBenchmark(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: TrackingBenchmark
        :param benchmark2: TrackingBenchmark
        :return:
        """
        if (not isinstance(benchmark1, tracking.TrackingBenchmark) or
                not isinstance(benchmark2, tracking.TrackingBenchmark)):
            self.fail('object was not a TrackingBenchmarknchmark')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.initializing_is_lost, benchmark2.initializing_is_lost)

    def test_benchmark_results_returns_a_benchmark_result(self):
        benchmark = tracking.TrackingBenchmark()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(self.trial_result.identifier, result.trial_result)

    def test_benchmark_produces_expected_results(self):
        # Perform the benchmark
        benchmark = tracking.TrackingBenchmark()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        # The expectations are based on the trajectory and tracking stats in setUp
        self.assertEqual(4, result.times_lost)
        self.assertEqual(1.3333, result.lost_intervals[0].start_time)
        self.assertEqual(1.6667, result.lost_intervals[0].end_time)
        self.assertEqual(1.6667 - 1.3333, result.lost_intervals[0].duration)
        self.assertEqual(10, result.lost_intervals[0].distance)
        self.assertEqual(1, result.lost_intervals[0].frames)

        self.assertEqual(2.3333, result.lost_intervals[1].start_time)
        self.assertEqual(3, result.lost_intervals[1].end_time)
        self.assertEqual(3 - 2.3333, result.lost_intervals[1].duration)
        self.assertEqual(20, result.lost_intervals[1].distance)
        self.assertEqual(2, result.lost_intervals[1].frames)

        self.assertEqual(3.3333, result.lost_intervals[2].start_time)
        self.assertEqual(4.3333, result.lost_intervals[2].end_time)
        self.assertAlmostEqual(1, result.lost_intervals[2].duration)
        self.assertEqual(30, result.lost_intervals[2].distance)
        self.assertEqual(3, result.lost_intervals[2].frames)

        self.assertEqual(4.6667, result.lost_intervals[3].start_time)
        self.assertEqual(4.6667, result.lost_intervals[3].end_time)
        self.assertEqual(0, result.lost_intervals[3].duration)
        self.assertEqual(0, result.lost_intervals[3].distance)
        self.assertEqual(1, result.lost_intervals[3].frames)
