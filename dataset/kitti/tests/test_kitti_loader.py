# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import pymongo.collection
import transforms3d as tf3d
import util.transform as tf
import database.client
import dataset.kitti.kitti_loader as kitti


class TestKITTILoader(unittest.TestCase):

    def test_make_camera_pose_returns_transform_object(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = kitti.make_camera_pose(frame_delta)
        self.assertIsInstance(pose, tf.Transform)

    def test_make_camera_pose_rearranges_location_coordinates(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = kitti.make_camera_pose(frame_delta)
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_make_camera_pose_changes_rotation_each_axis(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        # Roll, rotation around z-axis for kitti
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 0, 1), np.pi / 6, True)
        pose = kitti.make_camera_pose(frame_delta)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for kitti
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((1, 0, 0), np.pi / 6, True)
        pose = kitti.make_camera_pose(frame_delta)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for kitti
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 1, 0), np.pi / 6, True)
        pose = kitti.make_camera_pose(frame_delta)
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_make_camera_pose_combined(self):
        frame_delta = np.identity(4)
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            frame_delta[0:3, 3] = -loc[1], -loc[2], loc[0]
            frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((-rot_axis[1], -rot_axis[2], rot_axis[0]),
                                                              rot_angle, False)
            pose = kitti.make_camera_pose(frame_delta)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))

    def test_import_dataset_concrete(self):
        # TODO: Mock pykitti
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.image_collection.find_one.return_value = None
        mock_db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection)
        result = kitti.import_dataset('/storage/datasets/KITTI/dataset', mock_db_client)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))
