import unittest
import unittest.mock as mock
import numpy as np
import transforms3d as tf3d
import dataset.euroc.euroc_loader as euroc_loader
import util.transform as tf


class TestMakeCameraPose(unittest.TestCase):

    def test_location(self):
        forward = 51.2
        up = 153.3
        left = -126.07
        pose = euroc_loader.make_camera_pose(-1 * left, -1 * up, forward, 0, 0, 0, 1)
        self.assertNPEqual((forward, left, up), pose.location)

    def test_orientation(self):
        angle = np.pi / 7
        forward = 51.2
        up = 153.3
        left = -126.07
        quat = tf3d.quaternions.axangle2quat((-1 * left, -1 * up, forward), angle)
        pose = euroc_loader.make_camera_pose(0, 0, 0, quat[1], quat[2], quat[3], quat[0])
        self.assertNPEqual((0, 0, 0), pose.location)
        self.assertNPEqual(tf3d.quaternions.axangle2quat((forward, left, up), angle), pose.rotation_quat(True))

    def test_both(self):
        forward = 51.2
        up = 153.3
        left = -126.07
        angle = np.pi / 7
        o_forward = 1.151325
        o_left = 5.1315
        o_up = -0.2352323
        quat = tf3d.quaternions.axangle2quat((-1 * o_left, -1 * o_up, o_forward), angle)
        pose = euroc_loader.make_camera_pose(-1 * left, -1 * up, forward, quat[1], quat[2], quat[3], quat[0])
        self.assertNPEqual((forward, left, up), pose.location)
        self.assertNPEqual(tf3d.quaternions.axangle2quat((o_forward, o_left, o_up), angle), pose.rotation_quat(True))

    def test_randomized(self):
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            quat = tf3d.quaternions.axangle2quat((-rot_axis[1], -rot_axis[2], rot_axis[0]), rot_angle, False)
            pose = euroc_loader.make_camera_pose(-loc[1], -loc[2], loc[0], quat[1], quat[2], quat[3], quat[0])
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestReadTrajectory(unittest.TestCase):

    def test_reads_relative_to_first_pose(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = ""
        for time in np.arange(0, 10, 0.45):
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[time] = pose
            absolute_pose = first_pose.find_independent(pose)
            quat = absolute_pose.rotation_quat(w_first=True)
            trajectory_text += "{time},{x},{y},{z},{qw},{qx},{qy},{qz}\n".format(
                time=repr(time),
                x=repr(-1 * absolute_pose.location[1]),
                y=repr(-1 * absolute_pose.location[2]),
                z=repr(absolute_pose.location[0]),
                qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
            )
            self.assertEqual(time, float(repr(time)))

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath')
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def test_skips_comment_lines(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = ""
        line_template = "{time},{x},{y},{z},{qw},{qx},{qy},{qz}\n"
        for time in np.arange(0, 10, 0.45):
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[time] = pose
            absolute_pose = first_pose.find_independent(pose)
            quat = absolute_pose.rotation_quat(w_first=True)
            trajectory_text += line_template.format(
                time=repr(time),
                x=repr(-1 * absolute_pose.location[1]),
                y=repr(-1 * absolute_pose.location[2]),
                z=repr(absolute_pose.location[0]),
                qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
            )
            # Add incorrect trajectory data, preceeded by a hash to indicate it's a comment
            quat = pose.rotation_quat(w_first=True)
            trajectory_text += "# " + line_template.format(
                time=repr(time),
                x=repr(-1 * pose.location[1]),
                y=repr(-1 * pose.location[2]),
                z=repr(pose.location[0]),
                qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
            )
            self.assertEqual(time, float(repr(time)))

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath')
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(
            str([repr(f) for f in arr1]), str([repr(f) for f in arr2])))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def extend_mock_open(mock_open):
    """
    Extend the mock_open object to allow iteration over the file object.
    :param mock_open:
    :return:
    """
    handle = mock_open.return_value

    def _mock_file_iter():
        nonlocal handle
        for line in handle.readlines():
            yield line

    handle.__iter__.side_effect = _mock_file_iter
