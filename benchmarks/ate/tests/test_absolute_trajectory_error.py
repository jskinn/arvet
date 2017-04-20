import unittest
import numpy as np
import copy
import util.transform as tf
import util.associate as ass
import core.benchmark
import benchmarks.ate.absolute_trajectory_error as ate


def create_random_trajectory(random_state, duration=600, length=100):
    return {random_state.rand() * duration: tf.Transform(location=2000 * (random_state.rand(3) - 0.5),
                                                      rotation=random_state.rand(4))
            for _ in range(length)}


def rand_range(min_, max_, random_state):
    if hasattr(max_, 'shape'):
        shape = max_.shape
    else:
        shape = ()
    return min_ + (max_ - min_) * random_state.rand(*shape)


def create_noise(trajectory, random_state, time_offset=0, time_noise=0.01, loc_noise=100):
    if not isinstance(loc_noise, np.ndarray):
        loc_noise = np.array([loc_noise, loc_noise, loc_noise])

    noise = {}
    for time, pose in trajectory.items():
        noise[time] = rand_range(-loc_noise, loc_noise, random_state)

    relative_frame = tf.Transform(location=2000 * random_state.rand(3) - 1000,
                                  rotation=random_state.rand(4))

    changed_trajectory = {}
    for time, pose in trajectory.items():
        relative_pose = relative_frame.find_relative(pose)
        changed_trajectory[time + time_offset + rand_range(-time_noise, time_noise, random_state)] = tf.Transform(
            location=relative_pose.location + noise[time],
            rotation=relative_pose.rotation_quat(w_first=True),
            w_first=True)

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
        benchmark = ate.BenchmarkATE()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertEquals(benchmark.identifier, result.benchmark)
        self.assertEquals(self.trial_result.identifier, result.trial_result)

    def test_benchmark_results_fails_for_no_matching_timestaps(self):
        # Create two trajectories with no matching keys
        gt_traj = create_random_trajectory(self.random, 600, 10)
        comp_traj = create_random_trajectory(self.random, 600, 10)
        matches = ass.associate(gt_traj, comp_traj, offset=0, max_difference=0.02)
        for gt_key, comp_key in matches:
            del gt_traj[gt_key]
            del comp_traj[comp_key]

        # Set them as the inputs to the bechmark
        self.trial_result.ground_truth_trajectory = gt_traj
        self.trial_result.computed_trajectory = comp_traj

        # Perform the benchmark
        benchmark = ate.BenchmarkATE()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

    def test_benchmark_results_estimates_no_noise_for_identical_trajectory(self):
        # Copy the ground truth exactly
        self.trial_result.computed_trajectory = copy.deepcopy(self.trial_result.ground_truth_trajectory)

        benchmark = ate.BenchmarkATE()
        result = benchmark.benchmark_results(self.trial_result)

        for time, error in result.translational_error.items():
            self.assertAlmostEquals(0, error)

    def test_benchmark_results_estimates_no_noise_for_noiseless_trajectory(self):
        # Create a new computed trajectory with no noise
        comp_traj, noise = create_noise(self.trial_result.ground_truth_trajectory,
                                        self.random,
                                        time_offset=0,
                                        time_noise=0,
                                        loc_noise=0)
        self.trial_result.computed_trajectory = comp_traj

        benchmark = ate.BenchmarkATE()
        result = benchmark.benchmark_results(self.trial_result)

        for time, error in result.translational_error.items():
            self.assertAlmostEquals(0, error)

    def test_benchmark_results_estimates_reasonable_trajectory_noise(self):
        benchmark = ate.BenchmarkATE()
        result = benchmark.benchmark_results(self.trial_result)

        # The benchmark will not estimate the injected noise exactly.
        # Instead, we just want to assert that the difference between the noise used to create the trajectory
        # and the extracted error are a reasonable size
        noise_array = []
        error_array = []
        for time, error in result.translational_error.items():
            error_array.append(error)
            noise_array.append(self.noise[time][0:3])

        noise_array = np.matrix(noise_array).transpose()
        average_noise = np.sqrt(np.sum(np.multiply(noise_array, noise_array), 0)).A[0]   # Taken from the ATE code

        # Find the difference between the estimated error and the noise we added
        diff = average_noise - np.array(error_array)
        frac_diff = diff / average_noise

        self.assertLess(np.mean(frac_diff), 0.1)

    def test_offset_shifts_query_trajectory_time(self):
        # Create a new noise trajectory with a large time offset
        comp_traj, noise = create_noise(self.trial_result.ground_truth_trajectory, self.random, time_offset=1000)
        self.trial_result.computed_trajectory = comp_traj

        # This should fail due to the offset
        benchmark = ate.BenchmarkATE()
        result = benchmark.benchmark_results(self.trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

        # This one should work, since the offset brings things back close together
        benchmark.offset = -1000
        result = benchmark.benchmark_results(self.trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)

    def test_scale_affects_trajectory_position(self):
        # Manually scale the computed trajectory
        scale = 443
        for key, pose in self.trial_result.computed_trajectory.items():
            self.trial_result.computed_trajectory[key] = tf.Transform(location=pose.location / scale,
                                                                      rotation=pose.rotation_quat(True), w_first=True)

        # This should have a large error due to the bad scale
        benchmark = ate.BenchmarkATE()
        unscaled_result = benchmark.benchmark_results(self.trial_result)

        # This one should have a more reasonable error
        benchmark.scale = scale
        result = benchmark.benchmark_results(self.trial_result)
        self.assertLess(result.max, unscaled_result.max)
