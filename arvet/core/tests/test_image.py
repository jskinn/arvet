# Copyright (c) 2017, John Skinner
import os
import unittest
import numpy as np
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.metadata.tests.test_image_metadata import make_metadata
import arvet.util.test_helpers as th
import arvet.util.transform as tf
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import arvet.core.image as im


class TestImageDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        im.Image._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        im.Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads_simple(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        img.save()

        # Load all the entities
        all_entities = list(im.Image.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], img)
        all_entities[0].delete()

    def test_stores_and_loads_large(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=make_metadata(),
            additional_metadata={'test': True},
            depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            ground_truth_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
        )
        img.save()

        # Load all the entities
        all_entities = list(im.Image.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], img)
        all_entities[0].delete()


class TestImage(th.ExtendedTestCase):

    def test_camera_pose(self):
        pose = tf.Transform((13, -22, 43), (1, 2, -4, 2))
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC,
                camera_pose=pose
            )
        )
        self.assertNPEqual(img.camera_location, pose.location)
        self.assertNPEqual(img.camera_orientation, pose.rotation_quat(False))
        self.assertNPEqual(img.camera_pose, pose)
        self.assertNPEqual(img.camera_transform_matrix, pose.transform_matrix)

    def test_camera_pose_none(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.camera_location)
        self.assertIsNone(img.camera_orientation)
        self.assertIsNone(img.camera_pose)
        self.assertIsNone(img.camera_transform_matrix)


class TestStereoImageDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        im.Image._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        im.Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads_simple(self):
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            ),
            right_metadata = imeta.ImageMetadata(
                img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        img.save()

        # Load all the entities
        all_entities = list(im.Image.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], img)
        all_entities[0].delete()

    def test_stores_and_loads_large(self):
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=make_metadata(),
            right_metadata=make_metadata(img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0'),
            additional_metadata={'test': True},
            depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            right_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            ground_truth_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            right_ground_truth_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
        )
        img.save()

        # Load all the entities
        all_entities = list(im.Image.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], img)
        all_entities[0].delete()


class TestStereoImage(th.ExtendedTestCase):

    def setUp(self):
        self.left_pose = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        self.right_pose = tf.Transform(location=self.left_pose.find_independent((0, 0, 15)),
                                       rotation=self.left_pose.rotation_quat(w_first=False),
                                       w_first=False)
        self.metadata = imeta.ImageMetadata(
            img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=self.left_pose,
            intrinsics=cam_intr.CameraIntrinsics(32, 32, 17, 22, 16, 16),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=(
                imeta.LabelledObject(class_names=('car',), x=12, y=144, width=67, height=43,
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)), instance_name='Car-002'),
                imeta.LabelledObject(class_names=('cat',), x=125, y=244, width=117, height=67,
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     instance_name='cat-090')
            ))

        self.left_pixels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.right_pixels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.image = im.StereoImage(pixels=self.left_pixels,
                                    right_pixels=self.right_pixels,
                                    metadata=self.metadata,
                                    right_metadata=make_metadata(
                                        camera_pose=self.right_pose,
                                        intrinsics=cam_intr.CameraIntrinsics(32, 32, 8, 12, 16, 16)))

        self.full_left_pose = tf.Transform((4, 5, 6), (-0.5, 0.5, -0.5, 0.5))
        self.full_right_pose = tf.Transform(location=self.left_pose.find_independent((0, 0, 15)),
                                            rotation=self.left_pose.rotation_quat(w_first=False),
                                            w_first=False)
        self.full_metadata = imeta.ImageMetadata(
            img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=self.full_left_pose,
            intrinsics=cam_intr.CameraIntrinsics(32, 32, 17, 22, 16, 16),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=(
                imeta.LabelledObject(class_names=('car',), x=12, y=144, width=67, height=43,
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)),
                                     instance_name='Car-002'),
                imeta.LabelledObject(class_names=('cat',), x=125, y=244, width=117, height=67,
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     instance_name='cat-090')
            ))
        self.full_left_pixels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_right_pixels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.left_gt_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.right_gt_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.left_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.right_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.left_normals = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.right_normals = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_image = im.StereoImage(
            pixels=self.full_left_pixels,
            right_pixels=self.full_right_pixels,
            depth=self.left_depth,
            right_depth=self.right_depth,
            ground_truth_depth=self.left_gt_depth,
            right_ground_truth_depth=self.right_gt_depth,
            normals=self.left_normals,
            right_normals=self.right_normals,
            metadata=self.full_metadata,
            right_metadata=make_metadata(
                camera_pose=self.right_pose,
                intrinsics=cam_intr.CameraIntrinsics(32, 32, 8, 12, 16, 16)),
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        )

    def test_data(self):
        self.assertNPEqual(self.image.left_pixels, self.left_pixels)
        self.assertNPEqual(self.image.right_pixels, self.right_pixels)
        self.assertNPEqual(self.full_image.left_pixels, self.full_left_pixels)
        self.assertNPEqual(self.full_image.right_pixels, self.full_right_pixels)

    def test_camera_location(self):
        self.assertNPEqual(self.image.left_camera_location, np.array([1, 2, 3]))
        self.assertNPEqual(self.image.right_camera_location, self.right_pose.location)
        self.assertNPEqual(self.full_image.left_camera_location, np.array([4, 5, 6]))
        self.assertNPEqual(self.full_image.right_camera_location, self.full_right_pose.location)

    def test_camera_orientation(self):
        self.assertNPEqual(self.image.left_camera_orientation, np.array([0.5, 0.5, -0.5, -0.5]))
        self.assertNPEqual(self.image.right_camera_orientation, self.right_pose.rotation_quat(w_first=False))
        self.assertNPEqual(self.full_image.left_camera_orientation, np.array([-0.5, 0.5, -0.5, 0.5]))
        self.assertNPEqual(self.full_image.right_camera_orientation, self.full_right_pose.rotation_quat(w_first=False))

    def test_depth_data(self):
        self.assertEqual(self.image.left_depth, None)
        self.assertEqual(self.image.right_depth, None)
        self.assertNPEqual(self.full_image.left_depth, self.left_depth)
        self.assertNPEqual(self.full_image.right_depth, self.right_depth)

    def test_ground_truth_depth_data(self):
        self.assertEqual(self.image.left_ground_truth_depth, None)
        self.assertEqual(self.image.right_ground_truth_depth, None)
        self.assertNPEqual(self.full_image.left_ground_truth_depth, self.left_gt_depth)
        self.assertNPEqual(self.full_image.right_ground_truth_depth, self.right_gt_depth)

    def test_world_normals_data(self):
        self.assertEqual(self.image.left_normals, None)
        self.assertEqual(self.image.right_normals, None)
        self.assertNPEqual(self.full_image.left_normals, self.left_normals)
        self.assertNPEqual(self.full_image.right_normals, self.right_normals)

    def test_metadata(self):
        self.assertEqual(self.image.metadata, self.metadata)
        self.assertEqual(self.full_image.metadata, self.full_metadata)

    def test_additional_metadata(self):
        self.assertEqual(self.image.additional_metadata, {})
        self.assertEqual(self.full_image.additional_metadata, {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        })

    def test_left_image_is_base(self):
        self.assertNPEqual(self.image.pixels, self.image.left_pixels)
        self.assertNPEqual(self.image.camera_location, self.image.left_camera_location)
        self.assertNPEqual(self.image.camera_orientation, self.image.left_camera_orientation)
        self.assertNPEqual(self.image.depth, self.image.left_depth)
        self.assertNPEqual(self.image.normals, self.image.left_normals)

        self.assertNPEqual(self.full_image.pixels, self.full_image.left_pixels)
        self.assertNPEqual(self.full_image.camera_location, self.full_image.left_camera_location)
        self.assertNPEqual(self.full_image.camera_orientation, self.full_image.left_camera_orientation)
        self.assertNPEqual(self.full_image.depth, self.full_image.left_depth)
        self.assertNPEqual(self.full_image.normals, self.full_image.left_normals)

    def test_make_from_images(self):
        left_pose = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        right_pose = tf.Transform(location=left_pose.find_independent((0, 0, 15)),
                                  rotation=left_pose.rotation_quat(w_first=False),
                                  w_first=False)
        metadata = imeta.ImageMetadata(
            img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=left_pose,
            intrinsics=cam_intr.CameraIntrinsics(32, 32, 15, 21, 16, 16),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=(
                imeta.LabelledObject(class_names=('car',), x=12, y=144, width=67, height=43,
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)),
                                     instance_name='Car-002'),
                imeta.LabelledObject(class_names=('cat',), x=125, y=244, width=117, height=67,
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     instance_name='cat-090')
            ))
        right_metadata = make_metadata(camera_pose=right_pose,
                                       intrinsics=cam_intr.CameraIntrinsics(32, 32, 13, 7, 16, 16))
        left_image = im.Image(
            pixels=self.left_pixels,
            depth=self.left_depth,
            normals=self.left_normals,
            metadata=metadata,
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                },
                'reflection': 'missing'
            }
        )
        right_image = im.Image(
            pixels=self.right_pixels,
            depth=self.right_depth,
            normals=self.right_normals,
            metadata=right_metadata,
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 1,
                    'RoughnessQuality': False
                },
                'skeletons': 'There is already one inside you'
            }
        )

        stereo_image = im.StereoImage.make_from_images(left_image, right_image)
        self.assertEqual({
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 1,
                'RoughnessQuality': False
            },
            'skeletons': 'There is already one inside you',
            'reflection': 'missing'
        }, stereo_image.additional_metadata)
        self.assertNPEqual(stereo_image.left_camera_location, left_image.camera_location)
        self.assertNPEqual(stereo_image.left_camera_orientation, left_image.camera_orientation)
        self.assertNPEqual(stereo_image.left_pixels, left_image.pixels)
        self.assertNPEqual(stereo_image.left_depth, left_image.depth)
        self.assertNPEqual(stereo_image.left_normals, left_image.normals)

        self.assertNPEqual(stereo_image.right_camera_location, right_image.camera_location)
        self.assertNPEqual(stereo_image.right_camera_orientation, right_image.camera_orientation)
        self.assertNPEqual(stereo_image.right_pixels, right_image.pixels)
        self.assertNPEqual(stereo_image.right_depth, right_image.depth)
        self.assertNPEqual(stereo_image.right_normals, right_image.normals)
