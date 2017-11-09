# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import transforms3d as tf3d
import util.transform as tf
import dataset.tum.tum_loader as tum_loader


class TestTUMLoader(unittest.TestCase):

    def test_make_camera_pose_returns_transform_object(self):
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, 0, 0, 0, 1)
        self.assertIsInstance(pose, tf.Transform)

    def test_make_camera_pose_location_coordinates(self):
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, 0, 0, 0, 1)
        self.assertNPEqual((10, -22.4, 13.2), pose.location)

    def test_make_camera_pose_changes_rotation_each_axis(self):
        # Roll, rotation around x-axis?
        quat = tf3d.quaternions.axangle2quat((1, 0, 0), np.pi / 6)
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, quat[1], quat[2], quat[3], quat[0])
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around y-axis?
        quat = tf3d.quaternions.axangle2quat((0, 1, 0), np.pi / 6)
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, quat[1], quat[2], quat[3], quat[0])
        self.assertNPClose((0, np.pi / 6, 0), pose.euler)

        # Yaw, rotation around z-axis?
        quat = tf3d.quaternions.axangle2quat((0, 0, 1), np.pi / 6)
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, quat[1], quat[2], quat[3], quat[0])
        self.assertNPClose((0, 0, np.pi / 6), pose.euler)

    def test_make_camera_pose_combined(self):
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            quat = tf3d.quaternions.axangle2quat(rot_axis, rot_angle)
            pose = tum_loader.make_camera_pose(loc[0], loc[1], loc[2], quat[1], quat[2], quat[3], quat[0])
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))

    def test_associate_data_same_keys(self):
        desired_result = sorted(
            [np.random.uniform(0, 100),
             np.random.randint(0, 1000),
             np.random.uniform(-100, 100),
             "test-{0}".format(np.random.randint(0, 1000))]
        for _ in range(20))
        int_map = {stamp: int_val for stamp, int_val, _, _ in desired_result}
        float_map = {stamp: float_val for stamp, _, float_val, _ in desired_result}
        str_map = {stamp: str_val for stamp, _, _, str_val in desired_result}
        self.assertEqual(desired_result, tum_loader.associate_data(int_map, float_map, str_map))

    def test_associate_data_noisy_keys(self):
        random = np.random.RandomState()
        desired_result = sorted(
            [random.uniform(0, 100),
             random.randint(0, 1000),
             random.uniform(-100, 100),
             "test-{0}".format(random.randint(0, 1000))]
        for _ in range(20))
        int_map = {stamp: int_val for stamp, int_val, _, _ in desired_result}
        float_map = {stamp + random.uniform(-0.02, 0.02): float_val for stamp, _, float_val, _ in desired_result}
        str_map = {stamp + random.uniform(-0.02, 0.02): str_val for stamp, _, _, str_val in desired_result}
        self.assertEqual(desired_result, tum_loader.associate_data(int_map, float_map, str_map))

    def test_associate_data_missing_keys(self):
        random = np.random.RandomState()
        original_data = sorted(
            [idx / 2 + random.uniform(0, 0.01),
             random.randint(0, 1000),
             random.uniform(-100, 100),
             "test-{0}".format(random.randint(0, 1000))]
        for idx in range(20))
        int_map = {stamp: int_val for stamp, int_val, _, _ in original_data}
        float_map = {stamp + random.uniform(-0.02, 0.02): float_val for stamp, _, float_val, _ in original_data
                     if stamp > 2}
        str_map = {stamp + random.uniform(-0.02, 0.02): str_val for stamp, _, _, str_val in original_data
                   if stamp < 8}
        self.assertEqual([inner for inner in original_data if inner[0] > 2 and inner[0] < 8],
                         tum_loader.associate_data(int_map, float_map, str_map))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))
