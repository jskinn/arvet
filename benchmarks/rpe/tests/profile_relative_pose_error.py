#Copyright (c) 2017, John Skinner
import numpy as np
import transforms3d as tf3d
import util.transform as tf
import benchmarks.rpe.relative_pose_error as rpe

try:
    import cProfile as profile
except ImportError:
    import profile as profile


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


def main():
    random = np.random.RandomState(1311)  # Use a random stream to make the results consistent
    trajectory = create_random_trajectory(random)
    noisy_trajectory, noise = create_noise(trajectory, random)
    trial_result = MockTrialResult(gt_trajectory=trajectory, comp_trajectory=noisy_trajectory)

    benchmark = rpe.BenchmarkRPE()
    profile.runctx('benchmark.benchmark_results(trial_result)', globals=globals(), locals=locals(), sort='ncalls')


if __name__ == '__main__':
    main()
