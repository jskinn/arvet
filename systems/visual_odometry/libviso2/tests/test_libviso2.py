#Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import transforms3d as tf3d
import util.transform as tf
import util.dict_utils as du
import database.tests.test_entity
import systems.visual_odometry.libviso2.libviso2 as viso


class TestLibVisO(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return viso.LibVisOSystem

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'focal_distance': 700,
            'cu': 400,
            'cv': 300,
            'base': 0.6
        })
        return viso.LibVisOSystem(*args, **kwargs)

    def assert_models_equal(self, system1, system2):
        """
        Helper to assert that two viso systems are equal
        Libviso2 systems don't have any persistent configuration, so they're all equal with the same id.
        :param system1:
        :param system2:
        :return:
        """
        if (not isinstance(system1, viso.LibVisOSystem) or
                not isinstance(system2, viso.LibVisOSystem)):
            self.fail('object was not a LibVisOSystem')
        self.assertEqual(system1.identifier, system2.identifier)


class TestMakeRelativePose(unittest.TestCase):

    def test_returns_transform_object(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = viso.make_relative_pose(frame_delta)
        self.assertIsInstance(pose, tf.Transform)

    def test_rearranges_location_coordinates(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_changes_rotation_each_axis(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        # Roll, rotation around z-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 0, 1), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((1, 0, 0), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 1, 0), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_combined(self):
        frame_delta = np.identity(4)
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            frame_delta[0:3, 3] = -loc[1], -loc[2], loc[0]
            frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((-rot_axis[1], -rot_axis[2], rot_axis[0]),
                                                              rot_angle, False)
            pose = viso.make_relative_pose(frame_delta)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))
