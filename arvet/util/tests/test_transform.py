# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import transforms3d as tf3d
import arvet.util.transform as tf


def _make_quat(axis, theta):
    ax = np.asarray(axis)
    q = np.zeros(4)
    q[0] = np.cos(float(theta) / 2)
    q[1:] = np.sin(float(theta) / 2) * (ax / np.linalg.norm(ax))
    return q


class TestTransform(unittest.TestCase):

    def test_constructor_clone(self):
        trans1 = tf.Transform(location=(1, 2, 3),
                              rotation=(4, 5, 6, 7))
        trans2 = tf.Transform(trans1)
        self.assert_array(trans1.location, trans2.location)
        self.assert_array(trans1.rotation_quat(w_first=True), trans2.rotation_quat(w_first=True))
        self.assert_array(trans1.transform_matrix, trans2.transform_matrix)

    def test_location_basic(self):
        trans = tf.Transform(location=(1, 2, 3))
        self.assert_array((1, 2, 3), trans.location)

    def test_location_default(self):
        trans = tf.Transform()
        self.assert_array(np.zeros(3), trans.location)

    def test_constructor_location_from_homogeneous(self):
        hom = np.array([[0.80473785, -0.31061722, 0.50587936, 1],
                        [0.50587936, 0.80473785, -0.31061722, 2],
                        [-0.31061722, 0.50587936, 0.80473785, 3],
                        [0, 0, 0, 1]])
        trans = tf.Transform(hom)
        self.assert_array((1, 2, 3), trans.location)

    def test_constructor_rotation_basic(self):
        # The rotation here is for 45 degrees around axis 1,2,3
        trans = tf.Transform(location=(1, 2, 3), rotation=(0.92387953, 0.10227645, 0.2045529, 0.30682935), w_first=True)
        self.assert_close(trans.rotation_quat(w_first=True),
                          np.array([0.92387953, 0.10227645, 0.2045529, 0.30682935]))
        self.assert_close(trans.rotation_quat(w_first=False),
                          np.array([0.10227645, 0.2045529, 0.30682935, 0.92387953]))

    def test_constructor_rotation_handles_non_unit(self):
        trans = tf.Transform(rotation=(10, 1, 2, 3), w_first=True)
        self.assert_close(trans.rotation_quat(w_first=True), (0.93658581, 0.09365858, 0.18731716, 0.28097574))

    def test_constructor_rotation_default(self):
        trans = tf.Transform()
        self.assert_array(trans.rotation_quat(True), (1, 0, 0, 0))

    def test_constructor_euler_rotation(self):
        trans = tf.Transform(rotation=(np.pi / 6, np.pi / 4, np.pi / 3), w_first=True)
        self.assert_close(trans.euler, (np.pi / 6, np.pi / 4, np.pi / 3))
        self.assert_close(trans.euler, (np.pi / 6, np.pi / 4, np.pi / 3))

    def test_constructor_rotation_from_homogeneous(self):
        hom = np.array([[0.80473785, -0.31061722, 0.50587936, 1],
                        [0.50587936, 0.80473785, -0.31061722, 2],
                        [-0.31061722, 0.50587936, 0.80473785, 3],
                        [0, 0, 0, 1]])
        trans = tf.Transform(hom)
        self.assert_close(trans.rotation_quat(True), (0.92387953, 0.22094238, 0.22094238, 0.22094238))

    def test_rotation_matrix(self):
        trans = tf.Transform(rotation=(0.92387953, 0.22094238, 0.22094238, 0.22094238), w_first=True)
        self.assert_close(trans.rotation_matrix, np.array([[0.80473785, -0.31061722, 0.50587936],
                                                           [0.50587936, 0.80473785, -0.31061722],
                                                           [-0.31061722, 0.50587936, 0.80473785]]))

    def test_euler_each_axis(self):
        # Yaw
        qrot = _make_quat((0, 0, 1), np.pi / 6)
        trans = tf.Transform(rotation=qrot, w_first=True)
        self.assert_array(trans.euler, np.array([0, 0, np.pi / 6]))
        # Pitch
        qrot = _make_quat((0, 1, 0), np.pi / 6)
        trans = tf.Transform(rotation=qrot, w_first=True)
        self.assert_array(trans.euler, np.array([0, np.pi / 6, 0]))
        # Roll
        qrot = _make_quat((1, 0, 0), np.pi / 6)
        trans = tf.Transform(rotation=qrot, w_first=True)
        self.assert_array(trans.euler, np.array([np.pi / 6, 0, 0]))

    def test_equals(self):
        trans1 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans2 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans3 = tf.Transform(location=(1, 2, 4), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans4 = tf.Transform(location=(1, 2, 3), rotation=(0.5, -0.5, 0.5, -0.5))
        self.assertTrue(trans1 == trans1)
        self.assertTrue(trans1 == trans2)
        self.assertTrue(trans2 == trans1)
        self.assertEqual(trans1, trans2)
        self.assertFalse(trans1 == trans3)
        self.assertFalse(trans1 == trans4)
        self.assertFalse(trans3 == trans4)

    def test_not_equals(self):
        trans1 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans2 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans3 = tf.Transform(location=(1, 2, 4), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans4 = tf.Transform(location=(1, 2, 3), rotation=(0.5, -0.5, 0.5, -0.5))
        self.assertFalse(trans1 != trans1)
        self.assertFalse(trans1 != trans2)
        self.assertFalse(trans2 != trans1)
        self.assertTrue(trans1 != trans3)
        self.assertTrue(trans1 != trans4)
        self.assertTrue(trans3 != trans4)
        self.assertNotEqual(trans1, trans3)

    def test_equals_accepts_close_orientations(self):
        trans1 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans2 = tf.Transform(location=(1, 2, 3), rotation=(np.nextafter(-0.5, 1), 0.5, 0.5, -0.5))
        trans3 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, np.nextafter(0.5, 1), 0.5, -0.5))
        trans4 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, np.nextafter(0.5, 1), -0.5))
        trans5 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, np.nextafter(-0.5, 1)))
        self.assertTrue(trans1 == trans2)
        self.assertTrue(trans1 == trans3)
        self.assertTrue(trans1 == trans4)
        self.assertTrue(trans1 == trans5)

    def test_equals_accepts_inverted_orientations(self):
        trans1 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans2 = tf.Transform(location=(1, 2, 3), rotation=(0.5, -0.5, -0.5, 0.5))
        self.assertTrue(trans1 == trans2)

    def test_hash(self):
        trans1 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans2 = tf.Transform(location=(1, 2, 3), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans3 = tf.Transform(location=(1, 2, 4), rotation=(-0.5, 0.5, 0.5, -0.5))
        trans4 = tf.Transform(location=(1, 2, 3), rotation=(0.1, 0.2, 0.3, 0.4))
        self.assertEqual(hash(trans1), hash(trans1))
        self.assertEqual(hash(trans1), hash(trans2))
        self.assertNotEqual(hash(trans1), hash(trans3))
        self.assertNotEqual(hash(trans1), hash(trans4))
        self.assertEqual({trans1, trans1, trans1}, {trans1})    # Set literals

    def test_inverse_inverts_location(self):
        transform = tf.Transform(location=(10, 9, 8))
        inverse = transform.inverse()
        self.assert_array((-10, -9, -8), inverse.location)

    def test_inverse_inverts_rotation(self):
        quat = (-0.5, 0.5, 0.5, -0.5)
        q_inv = tf3d.quaternions.qinverse(quat)
        transform = tf.Transform(rotation=quat, w_first=True)
        inverse = transform.inverse()
        self.assert_array(q_inv, inverse.rotation_quat(w_first=True))

    def test_inverse_of_inverse_is_original(self):
        transform = tf.Transform(location=(13.2, -23.9, 5.5), rotation=_make_quat((1, 2, 4), np.pi / 7), w_first=True)
        inverse = transform.inverse()
        self.assertEqual(transform, inverse.inverse())

    def test_inverse_is_origin_relative_to_transform(self):
        transform = tf.Transform(location=(13.2, -23.9, 5.5), rotation=_make_quat((1, 2, 4), np.pi / 7), w_first=True)
        inverse = transform.inverse()
        self.assertEqual(transform.find_relative(tf.Transform()), inverse)

    def test_inverse_cancels_with_find_independent(self):
        transform = tf.Transform(location=(13.2, -23.9, 5.5), rotation=_make_quat((1, 2, 4), np.pi / 7), w_first=True)
        inverse = transform.inverse()
        self.assertEqual(tf.Transform(), transform.find_independent(inverse))
        self.assertEqual(tf.Transform(), inverse.find_independent(transform))

    def test_inverse_undoes_offset(self):
        transform_1 = tf.Transform(location=(13.2, -23.9, 5.5), rotation=_make_quat((1, 2, 4), np.pi / 7), w_first=True)
        offset = tf.Transform(location=(-6, 33.9, 8.2), rotation=_make_quat((6, 3, 0), 8 * np.pi / 19), w_first=True)
        transform_2 = transform_1.find_independent(offset)
        back_transform = transform_2.find_independent(offset.inverse())
        self.assert_close(transform_1.location, back_transform.location)
        self.assert_close(transform_1.rotation_quat(True), back_transform.rotation_quat(True))

    def test_find_relative_point_moves_origin(self):
        point = (11, 12, 13)
        trans = tf.Transform(location=(10, 9, 8))
        point_rel = trans.find_relative(point)
        self.assert_array(point_rel, (1, 3, 5))

    def test_find_relative_pose_moves_origin(self):
        pose = tf.Transform(location=(11, 12, 13))
        trans = tf.Transform(location=(10, 9, 8))
        pose_rel = trans.find_relative(pose)
        self.assert_array(pose_rel.location, (1, 3, 5))

    def test_find_relative_point_changes_location_coordinates(self):
        point = (11, 12, 13)
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        point_rel = trans.find_relative(point)
        self.assert_close(point_rel, (3, -1, 5))

    def test_find_relative_pose_changes_location_coordinates(self):
        pose = tf.Transform(location=(11, 12, 13))
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        pose_rel = trans.find_relative(pose)
        self.assert_close(pose_rel.location, (3, -1, 5))

    def test_find_relative_pose_changes_orientation(self):
        pose = tf.Transform(location=(11, 12, 13), rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        pose_rel = trans.find_relative(pose)
        self.assert_close(pose_rel.euler, (0, -np.pi / 4, -np.pi / 2))

    def test_find_relative_of_self_is_origin(self):
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        self.assertEqual(tf.Transform(), trans.find_relative(trans))

    def test_find_independent_point_moves_origin(self):
        point = (1, 3, 5)
        trans = tf.Transform(location=(10, 9, 8))
        point_rel = trans.find_independent(point)
        self.assert_array(point_rel, (11, 12, 13))

    def test_find_independent_pose_moves_origin(self):
        pose = tf.Transform(location=(1, 3, 5))
        trans = tf.Transform(location=(10, 9, 8))
        pose_rel = trans.find_independent(pose)
        self.assert_array(pose_rel.location, (11, 12, 13))

    def test_find_independent_pose_changes_orientation(self):
        pose = tf.Transform(location=(11, 12, 13), rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        pose_rel = trans.find_independent(pose)
        self.assert_close(pose_rel.euler, (0, np.pi / 4, np.pi / 2))

    def test_find_independent_of_origin_is_self(self):
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        self.assertEqual(trans, trans.find_independent(tf.Transform()))

    def test_find_relative_undoes_point(self):
        loc = (-13, 27, -127)
        qrot = _make_quat((-1, 0.1, -0.37), 7 * np.pi / 26)
        trans = tf.Transform(location=loc, rotation=qrot, w_first=True)

        point = (1, 2, 3)
        point_rel = trans.find_relative(point)
        point_prime = trans.find_independent(point_rel)
        self.assert_close(point_prime, point)

    def test_find_relative_undoes_pose(self):
        loc = (-13, 27, -127)
        qrot = _make_quat(np.array((-1, 0.1, -0.37)), 7 * np.pi / 26)
        trans = tf.Transform(location=loc, rotation=qrot, w_first=True)

        pose = tf.Transform(location=(10, 100, -5), rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        pose_rel = trans.find_relative(pose)
        pose_prime = trans.find_independent(pose_rel)

        self.assert_close(pose_prime.location, pose.location)
        self.assert_close(pose_prime.rotation_quat(w_first=True), pose.rotation_quat(w_first=True))

    def test_find_relative_preserves_equals(self):
        start_pose = tf.Transform(location=(11, 12, 13), rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        trans = tf.Transform(location=(10, 9, 8), rotation=_make_quat((0, 0, 1), np.pi / 2), w_first=True)
        pose = start_pose
        for _ in range(10):
            pose = trans.find_independent(trans.find_relative(pose))
            self.assertEqual(pose, start_pose)

    def test_relative_pose_contains_relative_point(self):
        loc = (-13, 27, -127)
        qrot = _make_quat(np.array((-1, 0.1, -0.37)), 7 * np.pi / 26)
        trans = tf.Transform(location=loc, rotation=qrot, w_first=True)

        point = np.array([41, -153, 16])
        pose = tf.Transform(location=point, rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        pose_rel = trans.find_relative(pose)
        point_rel = pose_rel.location
        point_prime = trans.find_independent(point_rel)

        self.assert_close(point, point_prime)

    def test_serialize_and_deserialise(self):
        random = np.random.RandomState(seed=1251)
        for _ in range(20):
            entity1 = tf.Transform(location=random.uniform(-1000, 1000, 3),
                                   rotation=random.uniform(-1, 1, 4), w_first=True)
            s_entity1 = entity1.serialize()

            entity2 = tf.Transform.deserialize(s_entity1)
            s_entity2 = entity2.serialize()
            self.assertEqual(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

            for idx in range(10):
                # Test that repeated serialization and deserialization does not degrade the information
                entity2 = tf.Transform.deserialize(s_entity2)
                s_entity2 = entity2.serialize()
                self.assertEqual(entity1, entity2)
                self.assertEqual(s_entity1, s_entity2)

    def test_forward_is_forward_vector(self):
        pose = tf.Transform(rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        self.assert_close((np.sqrt(2) / 2, 0, -np.sqrt(2) / 2), pose.forward)

        pose = tf.Transform(rotation=_make_quat((0, 0, 1), np.pi / 4), w_first=True)
        self.assert_close((np.sqrt(2) / 2, np.sqrt(2) / 2, 0), pose.forward)

    def test_forward_is_independent_of_location(self):
        quat = _make_quat((13, -4, 5), 13 * np.pi / 27)
        pose = tf.Transform(rotation=quat, w_first=True)
        forward = pose.forward
        for _ in range(10):
            pose = tf.Transform(location=np.random.uniform(-1000, 1000, 3), rotation=quat, w_first=True)
            self.assert_array(forward, pose.forward)

    def test_back_is_back_vector(self):
        pose = tf.Transform(rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        self.assert_close((-np.sqrt(2) / 2, 0, np.sqrt(2) / 2), pose.back)

        pose = tf.Transform(rotation=_make_quat((0, 0, 1), np.pi / 4), w_first=True)
        self.assert_close((-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0), pose.back)

    def test_back_is_independent_of_location(self):
        quat = _make_quat((13, -4, 5), 13 * np.pi / 27)
        pose = tf.Transform(rotation=quat, w_first=True)
        back = pose.back
        for _ in range(10):
            pose = tf.Transform(location=np.random.uniform(-1000, 1000, 3), rotation=quat, w_first=True)
            self.assert_array(back, pose.back)

    def test_left_is_left_vector(self):
        pose = tf.Transform(rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        self.assert_close((0, np.sqrt(2) / 2, np.sqrt(2) / 2), pose.left)

        pose = tf.Transform(rotation=_make_quat((0, 0, 1), np.pi / 4), w_first=True)
        self.assert_close((-np.sqrt(2) / 2, np.sqrt(2) / 2, 0), pose.left)

    def test_left_is_independent_of_location(self):
        quat = _make_quat((13, -4, 5), 13 * np.pi / 27)
        pose = tf.Transform(rotation=quat, w_first=True)
        left = pose.left
        for _ in range(10):
            pose = tf.Transform(location=np.random.uniform(-1000, 1000, 3), rotation=quat, w_first=True)
            self.assert_array(left, pose.left)

    def test_right_is_left_vector(self):
        pose = tf.Transform(rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        self.assert_close((0, -np.sqrt(2) / 2, -np.sqrt(2) / 2), pose.right)

        pose = tf.Transform(rotation=_make_quat((0, 0, 1), np.pi / 4), w_first=True)
        self.assert_close((np.sqrt(2) / 2, -np.sqrt(2) / 2, 0), pose.right)

    def test_right_is_independent_of_location(self):
        quat = _make_quat((13, -4, 5), 13 * np.pi / 27)
        pose = tf.Transform(rotation=quat, w_first=True)
        right = pose.right
        for _ in range(10):
            pose = tf.Transform(location=np.random.uniform(-1000, 1000, 3), rotation=quat, w_first=True)
            self.assert_array(right, pose.right)

    def test_up_is_up_vector(self):
        pose = tf.Transform(rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        self.assert_close((0, -np.sqrt(2) / 2, np.sqrt(2) / 2), pose.up)

        pose = tf.Transform(rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        self.assert_close((np.sqrt(2) / 2, 0, np.sqrt(2) / 2), pose.up)

    def test_up_is_independent_of_location(self):
        quat = _make_quat((13, -4, 5), 13 * np.pi / 27)
        pose = tf.Transform(rotation=quat, w_first=True)
        up = pose.up
        for _ in range(10):
            pose = tf.Transform(location=np.random.uniform(-1000, 1000, 3), rotation=quat, w_first=True)
            self.assert_array(up, pose.up)

    def test_down_is_down_vector(self):
        pose = tf.Transform(rotation=_make_quat((1, 0, 0), np.pi / 4), w_first=True)
        self.assert_close((0, np.sqrt(2) / 2, -np.sqrt(2) / 2), pose.down)

        pose = tf.Transform(rotation=_make_quat((0, 1, 0), np.pi / 4), w_first=True)
        self.assert_close((-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2), pose.down)

    def test_down_is_independent_of_location(self):
        quat = _make_quat((13, -4, 5), 13 * np.pi / 27)
        pose = tf.Transform(rotation=quat, w_first=True)
        down = pose.down
        for _ in range(10):
            pose = tf.Transform(location=np.random.uniform(-1000, 1000, 3), rotation=quat, w_first=True)
            self.assert_array(down, pose.down)

    def test_robust_normalize_sets_the_norm_to_approximately_1(self):
        for _ in range(100):
            x = np.random.uniform(-1, 1, 4)
            normed_x = tf.robust_normalize(x)
            self.assertTrue(np.isclose([1.0], [np.linalg.norm(normed_x)], atol=1e-14))

    def test_robust_normalize_doesnt_change_values_where_norm_is_already_1(self):
        for _ in range(100):
            x = np.random.uniform(-1, 1, 4)
            x = tf.robust_normalize(x)
            normed_x = tf.robust_normalize(x)
            self.assert_array(x, normed_x)

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


class TestHelpers(unittest.TestCase):

    def test_robust_normalize_normalizes_vector(self):
        vector = np.array([1, 2, 3])
        result = tf.robust_normalize(vector)
        self.assertAlmostEqual(1.0, np.linalg.norm(result), places=14)

    def test_robust_normalize_doesnt_change_already_unit_vectors(self):
        random = np.random.RandomState()
        for _ in range(100):
            vector = random.uniform(-1, 1, 4)
            norm_vector1 = tf.robust_normalize(vector)
            norm_vector2 = tf.robust_normalize(norm_vector1)
            self.assertTrue(np.array_equal(norm_vector1, norm_vector2))

    def test_quat_angle(self):
        theta = 1.32234252
        quat = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), theta)
        result = tf.quat_angle(quat)
        self.assertEqual(theta, result)

    def test_quat_diff_finds_simple_angle_between(self):
        quat1 = np.array([1, 0, 0, 0])
        quat2 = tf3d.quaternions.axangle2quat(np.array([0, 0, 1]), np.pi / 6)
        result = tf.quat_diff(quat1, quat2)
        self.assertAlmostEqual(np.pi / 6, result)

    def test_quat_diff_finds_angle_between(self):
        theta = 0.32234252  # Less than pi on 4
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), theta)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -theta)
        result = tf.quat_diff(quat1, quat2)
        self.assertAlmostEqual(2 * theta, result)

    def test_quat_diff_finds_angle_less_than_180deg(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), 2 * np.pi / 3)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -2 * np.pi / 3)
        result = tf.quat_diff(quat1, quat2)
        self.assertAlmostEqual(2 * np.pi / 3, result)

    def test_quat_diff_handles_handedness(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -np.pi / 6)
        result1 = tf.quat_diff(quat1, quat2)
        result2 = tf.quat_diff(quat1, -1 * quat2)
        result3 = tf.quat_diff(-1 * quat1, quat2)
        result4 = tf.quat_diff(-1 * quat1, -1 * quat2)
        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)
        self.assertEqual(result1, result4)
        self.assertAlmostEqual(np.pi / 3, result1)
        self.assertAlmostEqual(np.pi / 3, result2)
        self.assertAlmostEqual(np.pi / 3, result3)
        self.assertAlmostEqual(np.pi / 3, result4)

    def test_quat_diff_is_reflexive(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -np.pi / 6)
        result1 = tf.quat_diff(quat1, quat2)
        result2 = tf.quat_diff(quat2, quat1)
        self.assertEqual(result1, result2)
        self.assertAlmostEqual(np.pi / 3, result1)
        self.assertAlmostEqual(np.pi / 3, result2)

    def test_quat_diff_exhaustive(self):
        for degrees in range(360):
            theta = np.pi * degrees / 180
            axis = np.random.uniform(-1, 1, 3)
            quat1 = tf3d.quaternions.axangle2quat(axis, theta)
            quat2 = tf3d.quaternions.axangle2quat(axis, -1 * theta)
            result1 = tf.quat_diff(quat1, quat2)
            result2 = tf.quat_diff(quat2, quat1)
            self.assertEqual(result1, result2)

            answer = 2 * theta
            answer -= (answer // (2 * np.pi)) * 2 * np.pi
            if answer > np.pi:
                answer = 2 * np.pi - answer
            self.assertAlmostEqual(answer, result1)
            self.assertAlmostEqual(answer, result2)

    def test_quat_mean_of_single_quat_is_that_quat(self):
        quat = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), 1.32234252)
        result = tf.quat_mean([quat])
        self.assertNPEqual(quat, result)

    def test_quat_mean_of_two_mirrored_axis_is_zero(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

        quat1 = tf3d.quaternions.axangle2quat(np.array([0, 1, 0]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([0, 1, 0]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

        quat1 = tf3d.quaternions.axangle2quat(np.array([0, 0, 1]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([0, 0, 1]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

        quat1 = tf3d.quaternions.axangle2quat(np.array([0, 1, 1]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([0, 1, 1]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

        quat1 = tf3d.quaternions.axangle2quat(np.array([1, 0, 1]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, 0, 1]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

        quat1 = tf3d.quaternions.axangle2quat(np.array([1, 1, 0]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, 1, 0]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

    def test_quat_mean_of_two_mirrored_is_zero(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        self.assertNPClose([1, 0, 0, 0], result)

    def test_quat_mean_of_two_is_halfway_between(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), np.pi / 3)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), -np.pi / 6)
        mean = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), np.pi / 4 - np.pi / 6)
        result = tf.quat_mean([quat1, quat2])
        angle1 = tf.quat_diff(result, quat1)
        angle2 = tf.quat_diff(result, quat2)
        self.assertEqual(angle1, angle2, "{0} != {1}".format(180 * angle1 / np.pi, 180 * angle2 / np.pi))
        self.assertNPClose(mean, result)

    def test_quat_mean_of_two_order_is_irrelevant(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), np.pi / 3)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, 0, 0]), -np.pi / 6)
        result1 = tf.quat_mean([quat1, quat2])
        result2 = tf.quat_mean([quat2, quat1])
        self.assertNPClose(result1, result2)

    def test_quat_mean_of_two_close_together_is_halfway_between(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -2 * np.pi / 3 + np.pi / 360)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -2 * np.pi / 3 - np.pi / 360)
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -2 * np.pi / 3)
        result = tf.quat_mean([quat1, quat2])
        angle1 = tf.quat_diff(result, quat1)
        angle2 = tf.quat_diff(result, quat2)
        self.assertEqual(angle1, angle2)
        self.assertNPClose(mean, result)

    def test_quat_mean_of_two_handles_handedness(self):
        quat1 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quat2 = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -np.pi / 6)
        result1 = tf.quat_mean([quat1, quat2])
        result2 = tf.quat_mean([quat1, -1 * quat2])
        result3 = tf.quat_mean([-1 * quat1, quat2])
        result4 = tf.quat_mean([-1 * quat1, -1 * quat2])
        self.assertNPClose(result1, result2)
        self.assertNPClose(result1, -1 * result3)
        self.assertNPClose(result1, -1 * result4)
        self.assertNPClose([1, 0, 0, 0], result1)
        self.assertNPClose([1, 0, 0, 0], result2)
        self.assertNPClose([-1, 0, 0, 0], result3)
        self.assertNPClose([-1, 0, 0, 0], result4)

    def test_quat_mean_of_many_symetric_quats(self):
        quaternions = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            axis = np.random.uniform(-1, 1, 3)
            quaternions.append(tf3d.quaternions.axangle2quat(axis, angle))
            quaternions.append(tf3d.quaternions.axangle2quat(axis, -1 * angle))
        result = tf.quat_mean(quaternions)
        self.assertNPClose([1, 0, 0, 0], result)

    def test_quat_mean_of_many_quats(self):
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            axis = np.random.uniform(-1, 1, 3)
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, angle)))
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, -1 * angle)))
        result = tf.quat_mean(quaternions)
        self.assertNPClose(mean, result)

    def test_quat_mean_of_many_two_widely_spaced(self):
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 36, np.pi / 36)
            axis = np.random.uniform(-1, 1, 3)
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, angle)))
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, -1 * angle)))

        quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat([-6, 3, 7], -6 * np.pi / 7)))
        quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat([-6, 3, 7], 6 * np.pi / 7)))
        result = tf.quat_mean(quaternions)
        self.assertNPClose(mean, result)

    def test_quat_mean_of_many_widely_spaced_quats(self):
        # This is one of the pathalogical cases where the orientations are every which way,
        # so the noise obscures the actual mean, even though the quats happen to be symmetric.
        # We expect the mean to fail in this case.
        random = np.random.RandomState(1331)
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = []
        for idx in range(10):
            angle = random.uniform(-np.pi, np.pi)
            axis = random.uniform(-1, 1, 3)
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, angle)))
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, -1 * angle)))
        result = tf.quat_mean(quaternions)
        # print("Result is {0} degrees from mean".format(180 * tf.quat_diff(mean, result) / np.pi))
        self.assertNotNPClose(mean, result)

    def test_quat_mean_of_many_identical_quats(self):
        quat = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = [quat for _ in range(10)]
        result = tf.quat_mean(quaternions)
        self.assertNPClose(quat, result)

        quat = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), -np.pi / 6)
        quaternions = [quat for _ in range(10)]
        result = tf.quat_mean(quaternions)
        self.assertNPClose(quat, result)

    def test_quat_mean_of_many_close_together_quats(self):
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 360, np.pi / 360)
            axis = np.random.uniform(-1, 1, 3)
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, angle)))
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, -1 * angle)))
        result = tf.quat_mean(quaternions)
        self.assertNPClose(mean, result)

    def test_quat_mean_ignores_handedness(self):
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            axis = np.random.uniform(-1, 1, 3)
            quaternions.append(np.random.choice([-1, 1]) *
                               tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, angle)))
            quaternions.append(np.random.choice([-1, 1]) *
                               tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, -1 * angle)))
        result = tf.quat_mean(quaternions)
        if np.dot(mean, result) < 0:
            result = -1 * result
        self.assertNPClose(mean, result)

    def test_quat_mean_ignores_order(self):
        mean = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        quaternions = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            axis = np.random.uniform(-1, 1, 3)
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, angle)))
            quaternions.append(tf3d.quaternions.qmult(mean, tf3d.quaternions.axangle2quat(axis, -1 * angle)))
        for _ in range(10):
            # Try many times with different orders, which should give the same answer
            np.random.shuffle(quaternions)
            result = tf.quat_mean(quaternions)
            self.assertNPClose(mean, result)

    def test_compute_average_pose_finds_average_location(self):
        centre = np.random.uniform(-1000, 1000, 3)
        locations = [centre + np.random.normal(0, 10, 3) for _ in range(10)]
        mean_pose = tf.compute_average_pose(tf.Transform(location=loc) for loc in locations)
        self.assertNPEqual(np.mean(locations, axis=0), mean_pose.location)

    def test_compute_average_pose_finds_average_orientation(self):
        centre = tf3d.quaternions.axangle2quat(np.array([1, -2, 3]), np.pi / 6)
        rotations = []
        for _ in range(10):
            angle = np.random.uniform(-np.pi / 360, np.pi / 360)
            axis = np.random.uniform(-1, 1, 3)
            rotations.append(tf3d.quaternions.qmult(centre, tf3d.quaternions.axangle2quat(axis, angle)))
            rotations.append(tf3d.quaternions.qmult(centre, tf3d.quaternions.axangle2quat(axis, -1 * angle)))
        mean_pose = tf.compute_average_pose(tf.Transform(rotation=rot, w_first=True) for rot in rotations)
        self.assertNPClose(tf.quat_mean(rotations), mean_pose.rotation_quat(w_first=True))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))

    def assertNotNPClose(self, arr1, arr2):
        self.assertFalse(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are close".format(str(arr1), str(arr2)))
