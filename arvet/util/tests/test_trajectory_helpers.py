import unittest
import typing
import numpy as np
import transforms3d as tf3d
import arvet.util.transform as tf
from arvet.util.test_helpers import ExtendedTestCase
import arvet.util.trajectory_helpers as th


class TestZeroTrajectory(ExtendedTestCase):

    def test_zero_trajectory_starts_result_at_zero(self):
        traj = create_trajectory(start_location=np.array([100, 200, 300]))
        result = th.zero_trajectory(traj)
        first_pose = result[min(result.keys())]
        self.assertEqual(tf.Transform(), first_pose)

    def test_zero_trajectory_keeps_same_timestamps(self):
        traj = create_trajectory(start_location=np.array([100, 200, 300]))
        result = th.zero_trajectory(traj)
        self.assertEqual(set(traj.keys()), set(result.keys()))

    def test_zero_trajectory_preserves_relative_motion(self):
        traj = create_trajectory(start_location=np.array([100, 200, 300]))
        original_motions = th.trajectory_to_motion_sequence(traj)
        result = th.zero_trajectory(traj)
        result_motions = th.trajectory_to_motion_sequence(result)
        self.assertEqual(set(original_motions.keys()), set(result_motions.keys()))
        for time in original_motions.keys():
            self.assertNPClose(original_motions[time].location, result_motions[time].location)
            self.assertNPClose(original_motions[time].rotation_quat(True), result_motions[time].rotation_quat(True))


class TestFindTrajectoryScale(unittest.TestCase):

    def test_returns_zero_for_empty_trajectory(self):
        self.assertEqual(0, th.find_trajectory_scale({}))

    def test_returns_zero_for_single_pose_trajectory(self):
        self.assertEqual(0, th.find_trajectory_scale({0: tf.Transform()}))

    def test_returns_speed_for_constant_speed(self):
        random = np.random.RandomState(16492)
        speed = random.uniform(10, 100)
        traj = {
            float(time): tf.Transform(location=(speed * time, 0, 0))
            for time in range(100)
        }
        result = th.find_trajectory_scale(traj)
        self.assertAlmostEqual(speed, result, places=13)

    def test_returns_mean_speed(self):
        random = np.random.RandomState(56634)
        speeds = random.uniform(10, 100, 100)
        traj = {0: tf.Transform()}
        prev_location = np.zeros(3)
        for time in range(speeds.shape[0]):
            new_location = prev_location + np.array([speeds[time], 0, 0])
            traj[float(time) + 1] = tf.Transform(location=new_location)
            prev_location = new_location
        result = th.find_trajectory_scale(traj)
        self.assertAlmostEqual(float(np.mean(speeds)), result, places=13)

    def test_is_independent_of_direction_or_orientation(self):
        random = np.random.RandomState(15646)
        speeds = random.uniform(10, 100, 100)
        traj1 = {0: tf.Transform()}
        traj2 = {0: tf.Transform()}
        prev_location = np.zeros(3)
        for time in range(speeds.shape[0]):
            direction = random.uniform(-1, 1, 3)
            direction = direction / np.linalg.norm(direction)
            new_location = prev_location + speeds[time] * direction
            traj1[float(time) + 1] = tf.Transform(location=new_location)
            traj2[float(time) + 1] = tf.Transform(location=new_location, rotation=random.uniform(-1, 1, 4))
            prev_location = new_location
        no_motion_ref = th.find_trajectory_scale(traj1)
        result = th.find_trajectory_scale(traj2)
        self.assertAlmostEqual(float(np.mean(speeds)), result, places=13)
        self.assertEqual(no_motion_ref, result)

    def test_scales_with_time(self):
        random = np.random.RandomState(21273)
        speeds = random.uniform(10, 100, 100)
        times = np.abs(random.uniform(0.1, 3, 100))
        traj = {0: tf.Transform()}
        prev_location = np.zeros(3)
        prev_time = 0
        for idx in range(speeds.shape[0]):
            direction = random.uniform(-1, 1, 3)
            direction = direction / np.linalg.norm(direction)
            new_location = prev_location + speeds[idx] * times[idx] * direction
            traj[prev_time + times[idx]] = tf.Transform(location=new_location, rotation=random.uniform(-1, 1, 4))
            prev_location = new_location
            prev_time = prev_time + times[idx]
        result = th.find_trajectory_scale(traj)
        self.assertAlmostEqual(float(np.mean(speeds)), result, places=13)

    def test_handles_none(self):
        traj = {
            0: tf.Transform(),
            1: tf.Transform(location=(10, 0, 0)),
            2: None,
            3: tf.Transform(location=(30, 0, 0)),
            4: tf.Transform(location=(40, 0, 0)),
            5: None,
            6: tf.Transform(location=(60, 0, 0)),
            7: None
        }
        self.assertEqual(10, th.find_trajectory_scale(traj))

    def test_returns_zero_for_trajectory_entirely_none(self):
        result = th.find_trajectory_scale({0.334 * idx: None for idx in range(100)})
        self.assertEqual(0, result)


class TestRescaleTrajectory(ExtendedTestCase):

    def test_does_nothing_to_empty_trajectory(self):
        self.assertEqual({}, th.rescale_trajectory({}, 3))

    def test_does_nothing_to_single_pose_trajectory(self):
        self.assertEqual({0: tf.Transform()}, th.rescale_trajectory({0: tf.Transform()}, 3))

    def test_does_nothing_to_a_trajectory_entirely_none(self):
        traj = {0.2233 * idx: None for idx in range(10)}
        result = th.rescale_trajectory(traj, 10)
        self.assertEqual(traj, result)

    def test_changes_trajectory_scale(self):
        traj = create_trajectory(seed=64075)
        result = th.rescale_trajectory(traj, 10)
        self.assertAlmostEqual(10, th.find_trajectory_scale(result), places=14)

    def test_preserves_timestamps(self):
        traj = create_trajectory(seed=55607)
        result = th.rescale_trajectory(traj, 10)
        self.assertEqual(set(traj.keys()), set(result.keys()))

    def test_preserves_motion_direction(self):
        traj = create_trajectory(seed=23377)
        base_motions = th.trajectory_to_motion_sequence(traj)
        result = th.rescale_trajectory(traj, 10)
        result_motions = th.trajectory_to_motion_sequence(result)

        for time in base_motions.keys():
            if not np.array_equal(base_motions[time].location, np.zeros(3)):
                base_direction = base_motions[time].location / np.linalg.norm(base_motions[time].location)
                result_direction = result_motions[time].location / np.linalg.norm(result_motions[time].location)
                self.assertNPClose(base_direction, result_direction)

    def test_preserves_orientations(self):
        traj = create_trajectory(seed=56604)
        result = th.rescale_trajectory(traj, 10)
        for time in traj.keys():
            self.assertNPEqual(traj[time].rotation_quat(True), result[time].rotation_quat(True))

    def test_scales_uniformly(self):
        traj = create_trajectory(seed=47243)
        base_motions = th.trajectory_to_motion_sequence(traj)
        base_distances = {
            time: np.linalg.norm(motion.location)
            for time, motion in base_motions.items()
        }
        result = th.rescale_trajectory(traj, 10)
        result_motions = th.trajectory_to_motion_sequence(result)
        result_distances = {
            time: np.linalg.norm(motion.location)
            for time, motion in result_motions.items()
        }

        # For each pair of times, asssert that their relative scale remains the same
        for time1 in base_motions.keys():
            for time2 in base_motions.keys():
                if time2 > time1:
                    self.assertAlmostEqual(
                        base_distances[time1] / base_distances[time2],
                        result_distances[time1] / result_distances[time2],
                        places=10
                    )

    def test_handles_trajectory_containing_none(self):
        traj = {
            0: tf.Transform(),
            1: tf.Transform(location=(10, 0, 0)),
            2: None,
            3: tf.Transform(location=(30, 0, 0)),
            4: tf.Transform(location=(40, 0, 0)),
            5: None,
            6: tf.Transform(location=(60, 0, 0)),
            7: None
        }
        result = th.rescale_trajectory(traj, 1)
        self.assertEqual({
            0: tf.Transform(),
            1: tf.Transform(location=(1, 0, 0)),
            2: None,
            3: tf.Transform(location=(3, 0, 0)),
            4: tf.Transform(location=(4, 0, 0)),
            5: None,
            6: tf.Transform(location=(6, 0, 0)),
            7: None
        }, result)


class TestTrajectoryToMotionSequence(unittest.TestCase):

    def test_works_on_empty_trajectory(self):
        self.assertEqual({}, th.trajectory_to_motion_sequence({}))

    def test_works_on_single_pose_trajectory(self):
        self.assertEqual({}, th.trajectory_to_motion_sequence({0: tf.Transform()}))

    def test_skips_first_timestamp_due_to_fencepost_error(self):
        traj = create_trajectory(seed=64577)
        motions = th.trajectory_to_motion_sequence(traj)
        self.assertEqual(len(traj) - 1, len(motions))
        traj_times = set(traj.keys())
        motion_times = set(motions.keys())
        self.assertNotIn(min(traj_times), motion_times)
        self.assertEqual({time for time in traj_times if time != min(traj_times)}, motion_times)

    def test_contains_sequence_of_relative_motions(self):
        traj = create_trajectory(seed=58690)
        motions = th.trajectory_to_motion_sequence(traj)
        prev_time = 0
        for time in sorted(motions.keys()):
            self.assertEqual(traj[prev_time].find_relative(traj[time]), motions[time])
            prev_time = time


class TestComputeAverageTrajectory(ExtendedTestCase):

    def test_produces_average_location(self):
        # this is 5 sequences of 10 3-vector locations
        locations = np.random.normal(0, 4, (10, 5, 3))
        locations += np.array([[[idx, 25 - idx * idx, 3]] for idx in range(10)])

        # Flatten across the 5 sequences, giving 10 3-vectors
        mean_locations = np.mean(locations, axis=1)

        # Find the mean trajectory
        trajectories = [
            {
                time: tf.Transform(location=locations[time, traj_idx, :])
                for time in range(10)
            }
            for traj_idx in range(5)
        ]
        mean_trajectory = th.compute_average_trajectory(trajectories)

        # Check the locations are averaged
        for time in range(10):
            self.assertIn(time, mean_trajectory)
            self.assertNPEqual(mean_locations[time], mean_trajectory[time].location)

    def test_produces_average_orientation(self):
        # this is 5 sequences of 10 quaternion orientations that are close together
        orientations = [
            [
                tf3d.quaternions.axangle2quat(
                    (1, time_idx, 3),
                    (time_idx + np.random.uniform(-0.1, 0.1)) * np.pi / 17
                )
                for _ in range(5)
            ]
            for time_idx in range(10)
        ]

        # Flatten across the 5 sequences, giving 10 quaternions
        # We use our custom quat_mean, because average orientations are hard
        mean_orientations = [
            tf.quat_mean(orientations_at_time)
            for orientations_at_time in orientations
        ]

        # Find the mean trajectory
        trajectories = [
            {
                time_idx: tf.Transform(location=(time_idx, 0, 0),
                                       rotation=orientations[time_idx][traj_idx], w_first=True)
                for time_idx in range(10)
            }
            for traj_idx in range(5)
        ]
        mean_trajectory = th.compute_average_trajectory(trajectories)

        # Check the locations are averaged
        for time_idx in range(10):
            self.assertIn(time_idx, mean_trajectory)
            self.assertNPEqual(mean_orientations[time_idx], mean_trajectory[time_idx].rotation_quat(w_first=True))

    def test_associates_on_median_times(self):
        # 5 sequences of times, each length 10, randomly varied by 0.05, which is as much variance as this will handle
        times = np.random.uniform(-0.05, 0.05, (10, 5))
        times += np.arange(10).reshape(10, 1)   # Make time increase linearly

        # Find the median times
        median_times = np.median(times, axis=1)

        # this is 5 sequences of 10 3-vector locations
        locations = np.random.normal(0, 4, (10, 5, 3))
        locations += np.array([[[idx, 25 - idx * idx, 3]] for idx in range(10)])

        # Flatten across the 5 sequences, giving 10 3-vectors
        mean_locations = np.mean(locations, axis=1)

        # Find the mean trajectory
        trajectories = [
            {
                times[time_idx, traj_idx]: tf.Transform(location=locations[time_idx, traj_idx, :])
                for time_idx in range(10)
            }
            for traj_idx in range(5)
        ]
        mean_trajectory = th.compute_average_trajectory(trajectories)

        # Check the locations are averaged
        for time_idx in range(10):
            self.assertIn(median_times[time_idx], mean_trajectory)
            self.assertNPEqual(mean_locations[time_idx], mean_trajectory[median_times[time_idx]].location)

    def test_handles_missing_poses(self):
        # Choose different times in different trajectories where the pose will be missing
        missing = [[(time_idx + 1) * (traj_idx + 1) % 12 == 0 for traj_idx in range(5)] for time_idx in range(10)]

        # this is 5 sequences of 10 3-vector locations
        locations = np.random.normal(0, 4, (10, 5, 3))
        locations += np.array([[[idx, 25 - idx * idx, 3]] for idx in range(10)])

        # Flatten across the 5 sequences, giving 10 3-vectors
        # We have to leave the missing poses out of the mean
        mean_locations = [
            np.mean([
                locations[time_idx, traj_idx, :]
                for traj_idx in range(5)
                if not missing[time_idx][traj_idx]
            ], axis=0)
            for time_idx in range(10)
        ]

        # Find the mean trajectory
        trajectories = [
            {
                time_idx: tf.Transform(location=locations[time_idx, traj_idx, :])
                for time_idx in range(10)
                if not missing[time_idx][traj_idx]
            }
            for traj_idx in range(5)
        ]
        mean_trajectory = th.compute_average_trajectory(trajectories)

        # Check the locations are averaged
        for time in range(10):
            self.assertIn(time, mean_trajectory)
            self.assertNPEqual(mean_locations[time], mean_trajectory[time].location)

    def test_handles_poses_being_none(self):
        # Choose different times in different trajectories where the pose will be missing
        lost_start = 3
        lost_end = 7
        missing = [[(time_idx + 1) * (traj_idx + 1) % 12 == 0 for traj_idx in range(5)] for time_idx in range(10)]

        # this is 5 sequences of 10 3-vector locations
        locations = np.random.normal(0, 4, (10, 5, 3))
        locations += np.array([[[idx, 25 - idx * idx, 3]] for idx in range(10)])

        # Flatten across the 5 sequences, giving 10 3-vectors
        # We have to leave the missing poses out of the mean
        mean_locations = [
            np.mean([
                locations[time_idx, traj_idx, :]
                for traj_idx in range(5)
                if not lost_start + traj_idx <= time_idx < lost_end
            ], axis=0)
            for time_idx in range(10)
        ]

        # Find the mean trajectory
        trajectories = [
            {
                time_idx: tf.Transform(location=locations[time_idx, traj_idx, :])
                if not lost_start + traj_idx <= time_idx < lost_end else None
                for time_idx in range(10)
            }
            for traj_idx in range(5)
        ]
        mean_trajectory = th.compute_average_trajectory(trajectories)

        # Check the locations are averaged
        for time in range(10):
            self.assertIn(time, mean_trajectory)
            self.assertNPEqual(mean_locations[time], mean_trajectory[time].location)

    def test_handles_all_poses_at_given_time_being_none(self):

        # this is 5 sequences of 9 3-vector locations
        locations = np.random.normal(0, 4, (9, 5, 3))
        locations += np.array([[[idx, 25 - idx * idx, 3]] for idx in range(9)])

        # Flatten across the 5 sequences, giving 10 3-vectors
        # We have to leave the missing poses out of the mean
        mean_locations = [
            np.mean([
                locations[time_idx, traj_idx, :]
                for traj_idx in range(5)
            ], axis=0)
            for time_idx in range(9)
        ]

        # Find the mean trajectory
        trajectories = [
            {
                time_idx: tf.Transform(location=locations[time_idx - 1, traj_idx, :])
                if time_idx > 0 else None
                for time_idx in range(10)
            }
            for traj_idx in range(5)
        ]
        mean_trajectory = th.compute_average_trajectory(trajectories)

        # Check the locations are averaged
        for time_idx in range(10):
            self.assertIn(time_idx, mean_trajectory)
            if time_idx == 0:
                self.assertIsNone(mean_trajectory[time_idx])
            else:
                self.assertNPEqual(mean_locations[time_idx - 1], mean_trajectory[time_idx].location)


def create_trajectory(length=100, start_location=None, start_rotation=None, seed=0) -> \
        typing.Mapping[float, tf.Transform]:
    """
    Make a random trajectory with brownian motion, to do tests on
    :param length:
    :param start_location:
    :param start_rotation:
    :param seed:
    :return:
    """
    random = np.random.RandomState(seed)
    if start_location is None:
        start_location = np.zeros(3)
    if start_rotation is None:
        start_rotation = np.zeros(3)
    vel = np.zeros(3)
    angular_vel = np.zeros(3)
    time = 0
    traj = {0: tf.Transform(location=start_location, rotation=start_rotation)}
    for _ in range(length):
        next_time = time + random.uniform(0.1, 2.0)
        traj[next_time] = tf.Transform(
            location=traj[time].location + (next_time - time) * vel,
            rotation=traj[time].euler + (next_time - time) * angular_vel
        )
        time = next_time
        vel += random.normal(0, 1, 3)
        angular_vel += random.normal(-0.1, 0.1, 3)
    return traj
