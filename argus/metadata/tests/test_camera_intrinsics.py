# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import argus.metadata.camera_intrinsics as cam_intr


class TestImageMetadata(unittest.TestCase):

    def test_constructor_sets_properties(self):
        subject = cam_intr.CameraIntrinsics(
            width=800,
            height=600,
            fx=763.1,
            fy=759.2,
            cx=400,
            cy=300,
            skew=0.1,
            k1=0.01,
            k2=0.002,
            k3=0.2,
            p1=0.09,
            p2=-0.02)
        self.assertEqual(800, subject.width)
        self.assertEqual(600, subject.height)
        self.assertEqual(763.1, subject.fx)
        self.assertEqual(759.2, subject.fy)
        self.assertEqual(400, subject.cx)
        self.assertEqual(300, subject.cy)
        self.assertEqual(0.1, subject.s)
        self.assertEqual(0.01, subject.k1)
        self.assertEqual(0.002, subject.k2)
        self.assertEqual(0.2, subject.k3)
        self.assertEqual(0.09, subject.p1)
        self.assertEqual(-0.02, subject.p2)

    def test_equals(self):
        entity1 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(810, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 610, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 561.2, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 142.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 600, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 200, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.2, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.03, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.017, 0, 0.01, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, -0.19, -0.02)
        self.assertNotEqual(entity1, entity2)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, 0.8)
        self.assertNotEqual(entity1, entity2)

    def test_hash(self):
        entity1 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(810, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 610, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 561.2, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 142.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 600, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 200, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.2, 0.01, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.03, 0.002, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.017, 0, 0.01, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, -0.19, -0.02)
        self.assertNotEqual(hash(entity1), hash(entity2))
        entity2 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, 0.8)
        self.assertNotEqual(hash(entity1), hash(entity2))

    def test_serialize_and_deserialise(self):
        entity1 = cam_intr.CameraIntrinsics(800, 600, 763.1, 759.2, 400, 300, 0.1, 0.01, 0.002, 0, 0.01, -0.02)
        s_entity1 = entity1.serialize()

        entity2 = cam_intr.CameraIntrinsics.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assertEqual(entity1, entity2)
        self.assertEqual(s_entity1, s_entity2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = cam_intr.CameraIntrinsics.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assertEqual(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

    def test_serialize_and_deserialize_works_with_minimal_parameters(self):
        entity1 = cam_intr.CameraIntrinsics(800, 600, 513.2, 152.3, 400, 300)
        s_entity1 = entity1.serialize()

        entity2 = cam_intr.CameraIntrinsics.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assertEqual(entity1, entity2)
        self.assertEqual(s_entity1, s_entity2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = cam_intr.CameraIntrinsics.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assertEqual(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

    def test_horizontal_fov(self):
        rad_fov = 35.4898465 * np.pi / 180
        focal_length = 800 * 1 / (2 * np.tan(rad_fov / 2))
        subject = cam_intr.CameraIntrinsics(
            width=800, height=600, fx=focal_length, fy=focal_length, cx=400, cy=300)
        self.assertEqual(rad_fov, subject.horizontal_fov)

    def test_vertical_fov(self):
        rad_fov = 35.4898465 * np.pi / 180
        focal_length = 600 * 1 / (2 * np.tan(rad_fov / 2))
        subject = cam_intr.CameraIntrinsics(
            width=800, height=600, fx=focal_length, fy=focal_length, cx=400, cy=300)
        self.assertEqual(rad_fov, subject.vertical_fov)
