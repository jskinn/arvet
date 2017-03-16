import unittest
import numpy as np
import util.transform as trans


def _make_quat(axis, theta):
    ax = np.asarray(axis)
    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = np.sin(theta / 2) * (ax / np.linalg.norm(ax))
    return q


class TestTransform(unittest.TestCase):

    def test_constructor_clone(self):
        tf1 = trans.Transform(location=(1, 2, 3),
                             rotation=(4, 5, 6, 7))
        tf2 = trans.Transform(tf1)
        self.assert_array(tf1.location, tf2.location)
        self.assert_array(tf1.rotation_quat(w_first=True), tf2.rotation_quat(w_first=True))
        self.assert_array(tf1.transform_matrix, tf2.transform_matrix)

    def test_location_basic(self):
        tf = trans.Transform(location=(1, 2, 3))
        self.assert_array((1, 2, 3), tf.location)

    def test_location_default(self):
        tf = trans.Transform()
        self.assert_array(np.zeros(3), tf.location)

    def test_location_from_homogeneous(self):
        hom = np.array([[0.80473785, -0.31061722, 0.50587936, 1],
                        [0.50587936, 0.80473785, -0.31061722, 2],
                        [-0.31061722, 0.50587936, 0.80473785, 3],
                        [0, 0, 0, 1]])
        tf = trans.Transform(hom)
        self.assert_array((1, 2, 3), tf.location)

    def test_rotation_basic(self):
        # The rotation here is for 45 degrees around axis 1,2,3
        tf = trans.Transform(location=(1, 2, 3), rotation=(0.92387953, 0.10227645, 0.2045529, 0.30682935), w_first=True)
        self.assert_close(tf.rotation_quat(w_first=True),
                          np.array([0.92387953, 0.10227645, 0.2045529, 0.30682935]))
        self.assert_close(tf.rotation_quat(w_first=False),
                          np.array([0.10227645, 0.2045529, 0.30682935, 0.92387953]))

    def test_rotation_handles_non_unit(self):
        tf = trans.Transform(rotation=(10, 1, 2, 3), w_first=True)
        self.assert_close(tf.rotation_quat(w_first=True), (0.93658581, 0.09365858, 0.18731716, 0.28097574))

    def test_rotation_default(self):
        tf = trans.Transform()
        self.assert_array(tf.rotation_quat(True), (1, 0, 0, 0))

    def test_rotation_from_homogeneous(self):
        hom = np.array([[0.80473785, -0.31061722, 0.50587936, 1],
                        [0.50587936, 0.80473785, -0.31061722, 2],
                        [-0.31061722, 0.50587936, 0.80473785, 3],
                        [0, 0, 0, 1]])
        tf = trans.Transform(hom)
        self.assert_close(tf.rotation_quat(True), (0.92387953, 0.22094238, 0.22094238, 0.22094238))

    def test_euler_each_axis(self):
        # Pitch
        qrot = _make_quat((0, 0, 1), np.pi / 6)
        tf = trans.Transform(rotation=qrot, w_first=True)
        self.assert_array(tf.euler, np.array([np.pi / 6, 0, 0]))
        # Yaw
        qrot = _make_quat((0, 1, 0), np.pi / 6)
        tf = trans.Transform(rotation=qrot, w_first=True)
        self.assert_array(tf.euler, np.array([0, np.pi / 6, 0]))
        # Roll
        qrot = _make_quat((1, 0, 0), np.pi / 6)
        tf = trans.Transform(rotation=qrot, w_first=True)
        self.assert_array(tf.euler, np.array([0, 0, np.pi / 6]))

    def test_find_relative_point_moves_origin(self):
        point = (11, 12, 13)
        tf = trans.Transform(location=(10, 9, 8))
        point_rel = tf.find_relative(point)
        self.assert_array(point_rel, (1, 3, 5))

    def test_find_relative_pose_moves_origin(self):
        pose = trans.Transform(location=(11, 12, 13))
        tf = trans.Transform(location=(10, 9, 8))
        pose_rel = tf.find_relative(pose)
        self.assert_array(pose_rel.location, (1, 3, 5))

    def test_find_relative_point_changes_location_coordinates(self):
        point = (11, 12, 13)
        tf = trans.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        point_rel = tf.find_relative(point)
        self.assert_close(point_rel, (3, -1, 5))

    def test_find_relative_pose_changes_location_coordinates(self):
        pose = trans.Transform(location=(11, 12, 13))
        tf = trans.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        pose_rel = tf.find_relative(pose)
        self.assert_close(pose_rel.location, (3, -1, 5))

    def test_find_relative_pose_changes_orientation(self):
        pose = trans.Transform(location=(11, 12, 13), rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        tf = trans.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        pose_rel = tf.find_relative(pose)
        self.assert_close(pose_rel.euler, (-np.pi / 2, -np.pi / 4, 0))

    def test_find_independent_point_moves_origin(self):
        point = (1, 3, 5)
        tf = trans.Transform(location=(10, 9, 8))
        point_rel = tf.find_independent(point)
        self.assert_array(point_rel, (11, 12, 13))

    def test_find_independent_pose_moves_origin(self):
        pose = trans.Transform(location=(1, 3, 5))
        tf = trans.Transform(location=(10, 9, 8))
        pose_rel = tf.find_independent(pose)
        self.assert_array(pose_rel.location, (11, 12, 13))

    def test_find_relative_undoes_point(self):
        loc = (-13, 27, -127)
        qrot = _make_quat((-1, 0.1, -0.37), 7 * np.pi / 26)
        tf = trans.Transform(location=loc, rotation=qrot, w_first=True)

        point = (1, 2, 3)
        point_rel = tf.find_relative(point)
        point_prime = tf.find_independent(point_rel)
        self.assert_close(point_prime, point)

    def test_find_relative_undoes_pose(self):
        loc = (-13, 27, -127)
        qrot = _make_quat(np.array((-1, 0.1, -0.37)), 7 * np.pi / 26)
        tf = trans.Transform(location=loc, rotation=qrot, w_first=True)

        pose = trans.Transform(location=(10, 100, -5), rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        pose_rel = tf.find_relative(pose)
        pose_prime = tf.find_independent(pose_rel)

        self.assert_close(pose_prime.location, pose.location)
        self.assert_close(pose_prime.rotation_quat(w_first=True), pose.rotation_quat(w_first=True))

    def test_relative_pose_contains_relative_point(self):
        loc = (-13, 27, -127)
        qrot = _make_quat(np.array((-1, 0.1, -0.37)), 7 * np.pi / 26)
        tf = trans.Transform(location=loc, rotation=qrot, w_first=True)

        point = np.array([41, -153, 16])
        pose = trans.Transform(location=point, rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        pose_rel = tf.find_relative(pose)
        point_rel = pose_rel.location
        point_prime = tf.find_independent(point_rel)

        self.assert_close(point, point_prime)

    def assert_array(self, arr1, arr2, msg=None):
        a1 = np.asarray(arr1)
        a2 = np.asarray(arr2)
        if msg is None:
            msg = "{0} is not equal to {1}".format(str(a1), str(a2))
        self.assertTrue(np.array_equal(a1, a2), msg)

    def assert_close(self, arr1, arr2, msg=None):
        a1 = np.asarray(arr1)
        a2 = np.asarray(arr2)
        if msg is None:
            msg = "{0} is not close to {1}".format(str(a1), str(a2))
        self.assertTrue(np.all(np.isclose(a1, a2)), msg)
