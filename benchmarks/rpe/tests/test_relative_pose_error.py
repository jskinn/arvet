import unittest
import numpy as np
import copy
import transforms3d as tf3d
import util.transform as tf
import core.benchmark
import benchmarks.rpe.relative_pose_error as rpe


def create_random_trajectory(random_state, duration=600, length=100):
    return {random_state.uniform(0, duration):
            tf.Transform(location=random_state.uniform(-1000, 1000, 3), rotation=random_state.uniform(0, 1, 4))
            for _ in range(length)}


def create_noise(trajectory, random_state, time_offset=0, time_noise=0.01, loc_noise=100, rot_noise=np.pi/64):
    if not isinstance(loc_noise, np.ndarray):
        loc_noise = np.array([loc_noise, loc_noise, loc_noise])

    noise = {}
    for time, pose in trajectory.items():
        noise[time] = tf.Transform(location=random_state.uniform(-loc_noise, loc_noise),
                                   rotation=tf3d.quaternions.axangle2quat(random_state.uniform(-1, 1, 3),
                                                                          random_state.uniform(-rot_noise, rot_noise)),
                                   w_first=True)

    relative_frame = tf.Transform(location=random_state.uniform(-1000, 1000, 3),
                                  rotation=random_state.uniform(0, 1, 4))

    changed_trajectory = {}
    for time, pose in trajectory.items():
        relative_pose = relative_frame.find_relative(pose)
        noisy_time = time + time_offset + random_state.uniform(-time_noise, time_noise)
        noisy_pose = relative_pose.find_independent(noise[time])
        changed_trajectory[noisy_time] = noisy_pose

    return changed_trajectory, noise


class MockTrialResult:

    def __init__(self, gt_trajectory, comp_trajectory):
        self._gt_traj = gt_trajectory
        self._comp_traj = comp_trajectory

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
    def computed_trajectory(self):
        return self._comp_traj

    @computed_trajectory.setter
    def computed_trajectory(self, computed_trajectory):
        self._comp_traj = computed_trajectory

    def get_ground_truth_camera_poses(self):
        return self._gt_traj

    def get_computed_camera_poses(self):
        return self._comp_traj


class TestBenchmarkATE(unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1311)   # Use a random stream to make the results consistent
        trajectory = create_random_trajectory(self.random)
        noisy_trajectory, self.noise = create_noise(trajectory, self.random)
        self.trial_result = MockTrialResult(gt_trajectory=trajectory, comp_trajectory=noisy_trajectory)

    def test_benchmark_results_returns_a_benchmark_result(self):
        benchmark = rpe.BenchmarkRPE()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(self.trial_result.identifier, result.trial_result)

    def test_benchmark_results_fails_for_no_matching_timestaps(self):
        # Create a new computed trajectory with no matching timestamps
        self.trial_result.computed_trajectory = {time + 10000: pose
                                                 for time, pose in self.trial_result.ground_truth_trajectory.items()}

        # Perform the benchmark
        benchmark = rpe.BenchmarkRPE()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

    def test_benchmark_results_estimates_no_error_for_identical_trajectory(self):
        # Copy the ground truth exactly
        self.trial_result.computed_trajectory = copy.deepcopy(self.trial_result.ground_truth_trajectory)

        benchmark = rpe.BenchmarkRPE()
        result = benchmark.benchmark_results(self.trial_result)

        if isinstance(result, core.benchmark.FailedBenchmark):
            print(result.reason)

        for time, error in result.translational_error.items():
            self.assertAlmostEqual(0, error)
        for time, error in result.rotational_error.items():
            self.assertAlmostEqual(0, error)

    def test_benchmark_results_estimates_no_error_for_noiseless_trajectory(self):
        # Create a new computed trajectory with no noise
        comp_traj, noise = create_noise(self.trial_result.ground_truth_trajectory,
                                        self.random,
                                        time_offset=0,
                                        time_noise=0,
                                        loc_noise=0,
                                        rot_noise=0)
        self.trial_result.computed_trajectory = comp_traj

        benchmark = rpe.BenchmarkRPE()
        result = benchmark.benchmark_results(self.trial_result)

        for time, error in result.translational_error.items():
            self.assertAlmostEqual(0, error)
        for time, error in result.rotational_error.items():
            self.assertAlmostEqual(0, error)

    def test_benchmark_results_estimates_reasonable_trajectory_noise(self):
        benchmark = rpe.BenchmarkRPE()
        result = benchmark.benchmark_results(self.trial_result)

        # The benchmark will not estimate the injected noise exactly.
        # Instead, we just want to assert that the difference between the noise used to create the trajectory
        # and the extracted error are a reasonable size
        #TODO: No idea how to tell the error is reasonable

    def test_offset_shifts_query_trajectory_time(self):
        # Create a new noise trajectory with a large time offset
        comp_traj, noise = create_noise(self.trial_result.ground_truth_trajectory, self.random, time_offset=1000)
        self.trial_result.computed_trajectory = comp_traj

        # This should fail due to the offset
        benchmark = rpe.BenchmarkRPE()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

        # This one should work, since the offset brings things back close together
        benchmark.offset = -1000
        result = benchmark.benchmark_results(self.trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)

    def test_scale_affects_trajectory_position(self):
        # Manually scale the computed trajectory
        scale = 4243
        self.trial_result.computed_trajectory = {}
        for key, pose in self.trial_result.ground_truth_trajectory.items():
            self.trial_result.computed_trajectory[key] = tf.Transform(location=pose.location / scale,
                                                                      rotation=pose.rotation_quat(True), w_first=True)

        # This should have a large error due to the bad scale
        benchmark = rpe.BenchmarkRPE()
        unscaled_result = benchmark.benchmark_results(self.trial_result)

        # This one should have a more reasonable error
        benchmark.scale = scale
        result = benchmark.benchmark_results(self.trial_result)
        self.assertLess(result.trans_max, unscaled_result.trans_max)
        # We don't test rotation error, it isn't affected by scale
