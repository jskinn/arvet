from unittest import TestCase
from random import uniform
from numpy import array as nparray, array_equal
from core.trajectory import (Trajectory,
                             TrajectoryPoint,
                             TrajectoryBuilder,
                             tuples_to_trajectory,
                             trajectory_to_tuples,
                             serialize_trajectory,
                             deserialize_trajectory)


class TrajectoryTest(TestCase):

    def test_iterable(self):
        points = range(10)
        traj = Trajectory(points)
        self.assertTrue(hasattr(traj, '__iter__'))
        for point in traj:
            self.assertEqual(point, points[point])

    def test_get(self):
        points = range(10)
        traj = Trajectory(points)
        for point in points:
            self.assertEqual(point, traj.get(point))

    def test_len(self):
        points = range(10)
        traj = Trajectory(points)
        self.assertEqual(len(points), len(traj))


class TrajectoryPointTest(TestCase):

    def test_timestamp_cannot_be_negative(self):
        point = TrajectoryPoint(12, nparray([1, 2, 3]), nparray([4, 5, 6, 7]))
        point.timestamp = -21
        self.assertEqual(point.timestamp, 12)

    def test_location_is_reshaped(self):
        point = TrajectoryPoint(1, nparray([[1], [2], [3]]), nparray([4, 5, 6, 7]))
        self.assertTrue(array_equal(point.location, nparray([1,2,3])))
        point = TrajectoryPoint(1, nparray([[1, 2, 3]]), nparray([4, 5, 6, 7]))
        self.assertTrue(array_equal(point.location, nparray([1, 2, 3])))

    def test_default_location(self):
        point = TrajectoryPoint(1, [], nparray([4, 5, 6, 7]))
        self.assertTrue(array_equal(point.location, nparray([0, 0, 0])))

    def test_orientation_is_reshaped(self):
        point = TrajectoryPoint(1, nparray([4, 5, 6]), nparray([[1], [2], [3], [4]]))
        self.assertTrue(array_equal(point.orientation, nparray([1, 2, 3, 4])))
        point = TrajectoryPoint(1, nparray([4, 5, 6]), nparray([[1, 2, 3, 4]]), )
        self.assertTrue(array_equal(point.orientation, nparray([1, 2, 3, 4])))

    def test_default_orientation(self):
        point = TrajectoryPoint(1, nparray([4, 5, 6]), [])
        self.assertTrue(array_equal(point.orientation, nparray([0, 0, 0, 1])))


class TrajectoryBuilderTest(TestCase):

    def test_add_arguments(self):
        builder = TrajectoryBuilder()
        builder.add_arguments(1, 2, 3, 4, 5, 6, 7, 8)
        traj = builder.build()
        self.assertEqual(len(traj), 1)
        point = traj.get(0)
        self.assertEqual(point.timestamp, 1)
        self.assertTrue(array_equal(point.location, nparray([2, 3, 4])))
        self.assertTrue(array_equal(point.orientation, nparray([5, 6, 7, 8])))

    def test_add_numpy(self):
        builder = TrajectoryBuilder()
        builder.add_numpy(1, nparray([2, 3, 4]), nparray([5, 6, 7, 8]))
        traj = builder.build()
        self.assertEqual(len(traj), 1)
        point = traj.get(0)
        self.assertEqual(point.timestamp, 1)
        self.assertTrue(array_equal(point.location, nparray([2, 3, 4])))
        self.assertTrue(array_equal(point.orientation, nparray([5, 6, 7, 8])))


class TestBuilderHelpers(TestCase):

    def test_tuples_to_trajectory(self):
        # Build a random trajectory of tuples
        tuples = []
        for idx in range(200):
            tuples.append((uniform(0,100),  # Timestamp
                           uniform(-1000, 1000), uniform(-1000, 1000), uniform(-1000, 1000),    # location
                           uniform(-1000, 1000), uniform(-1000, 1000), uniform(-1000, 1000), uniform(-1000, 1000)))

        initial_traj = tuples_to_trajectory(tuples)
        for idx in range(0, len(tuples)):
            point = initial_traj.get(idx)
            self.assertEqual(point.timestamp, tuples[idx][0])
            self.assertTrue(array_equal(point.location, nparray([tuples[idx][1],
                                                                 tuples[idx][2],
                                                                 tuples[idx][3]])))
            self.assertTrue(array_equal(point.orientation, nparray([tuples[idx][4],
                                                                 tuples[idx][5],
                                                                 tuples[idx][6],
                                                                 tuples[idx][7]])))

    def test_trajectory_to_tuples(self):
        # Build a random trajectory of tuples
        tuples = []
        for idx in range(200):
            tuples.append((uniform(0, 100),  # Timestamp
                           uniform(-1000, 1000), uniform(-1000, 1000), uniform(-1000, 1000),  # location
                           uniform(-1000, 1000), uniform(-1000, 1000), uniform(-1000, 1000), uniform(-1000, 1000)))

        traj = tuples_to_trajectory(tuples)
        tuples2 = trajectory_to_tuples(traj)
        self.assertEqual(tuples, tuples2)
