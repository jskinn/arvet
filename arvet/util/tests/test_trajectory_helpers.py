import unittest
import typing
import numpy as np
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th


class TestZeroTrajectory(unittest.TestCase):

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

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestFindTrajectoryScale(unittest.TestCase):

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
        self.assertAlmostEqual(np.mean(speeds), result, places=13)

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
        self.assertAlmostEqual(np.mean(speeds), result, places=13)
        self.assertEqual(no_motion_ref, result)

    def test_scales_with_time(self):
        random = np.random.RandomState(21273)
        speeds = random.uniform(10, 100, 100)
        times = random.uniform(0.1, 3, 100)
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
        self.assertAlmostEqual(np.mean(speeds), result, places=13)


class TestRescaleTrajectory(unittest.TestCase):

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

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestTrajectoryToMotionSequence(unittest.TestCase):

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
