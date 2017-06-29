import unittest
import numpy as np
import util.transform as tf
import util.dict_utils as du
import metadata.image_metadata as imeta
import core.image as im


class TestImage(unittest.TestCase):

    def setUp(self):
        metadata = imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT,
            time_of_day=imeta.TimeOfDay.DAY,
            height=600,
            width=800,
            fov=90,
            focal_length=5,
            aperture=22,
            simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT,
            texture_mipmap_bias=1,
            normal_mipmap_bias=2,
            roughness_enabled=True,
            geometry_decimation=0.8,
            procedural_generation_seed=16234,
            labelled_objects=(
                imeta.LabelledObject(class_names=('car',), bounding_box=(12, 144, 67, 43), label_color=(123, 127, 112),
                                     relative_pose=tf.Transform((12,3,4), (0.5, 0.1, 1, 1.7)), object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',), bounding_box=(125, 244, 117, 67), label_color=(27, 89, 62),
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     object_id='cat-090')
            ),
            average_scene_depth=90.12)

        trans = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        self.image_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.image = im.Image(
            data=self.image_data,
            camera_pose=trans,
            metadata=metadata)

        trans = tf.Transform((4, 5, 6), (0.5, -0.5, 0.5, -0.5))
        self.full_image_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_image_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.full_image_labels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_image_normals = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_image = im.Image(
            data=self.full_image_data,
            camera_pose=trans,
            depth_data=self.full_image_depth,
            labels_data=self.full_image_labels,
            world_normals_data=self.full_image_normals,
            metadata=metadata,
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
        self.assertTrue(np.array_equal(self.image.data, self.image_data))
        self.assertTrue(np.array_equal(self.full_image.data, self.full_image_data))

    def test_camera_location(self):
        self.assertTrue(np.array_equal(self.image.camera_location, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(self.full_image.camera_location, np.array([4, 5, 6])))

    def test_camera_orientation(self):
        self.assertTrue(np.array_equal(self.image.camera_orientation, np.array([0.5, 0.5, -0.5, -0.5])))
        self.assertTrue(np.array_equal(self.full_image.camera_orientation, np.array([0.5, -0.5, 0.5, -0.5])))

    def test_depth_filename(self):
        self.assertEqual(self.image.depth_data, None)
        self.assertTrue(np.array_equal(self.full_image.depth_data, self.full_image_depth))

    def test_labels_filename(self):
        self.assertEqual(self.image.labels_data, None)
        self.assertTrue(np.array_equal(self.full_image.labels_data, self.full_image_labels))

    def test_world_normals_filename(self):
        self.assertEqual(self.image.world_normals_data, None)
        self.assertTrue(np.array_equal(self.full_image.world_normals_data, self.full_image_normals))

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

    def test_padded_kwargs(self):
        kwargs = {'a': 1, 'b': 2, 'c': 3}
        with self.assertRaises(TypeError):
            im.Image(13/30, np.random.uniform(0, 1, (32, 32)), np.array([1, 2, 3]), np.array([1, 2, 3, 4]), **kwargs)


class TestStereoImage(unittest.TestCase):

    def setUp(self):
        metadata = imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT,
            time_of_day=imeta.TimeOfDay.DAY,
            height=600,
            width=800,
            fov=90,
            focal_length=5,
            aperture=22,
            simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT,
            texture_mipmap_bias=1,
            normal_mipmap_bias=2,
            roughness_enabled=True,
            geometry_decimation=0.8,
            procedural_generation_seed=16234,
            labelled_objects=(
                imeta.LabelledObject(class_names=('car',), bounding_box=(12, 144, 67, 43), label_color=(123, 127, 112),
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)), object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',), bounding_box=(125, 244, 117, 67), label_color=(27, 89, 62),
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     object_id='cat-090')
            ),
            average_scene_depth=90.12)

        self.left_pose = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        self.right_pose = tf.Transform(location=self.left_pose.find_independent((0, 0, 15)),
                                       rotation=self.left_pose.rotation_quat(w_first=False),
                                       w_first=False)
        self.left_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.right_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.image = im.StereoImage(left_data=self.left_data,
                                    left_camera_pose=self.left_pose,
                                    right_data=self.right_data,
                                    right_camera_pose=self.right_pose,
                                    metadata=metadata)

        self.full_left_pose = tf.Transform((4, 5, 6), (-0.5, 0.5, -0.5, 0.5))
        self.full_right_pose = tf.Transform(location=self.left_pose.find_independent((0, 0, 15)),
                                            rotation=self.left_pose.rotation_quat(w_first=False),
                                            w_first=False)
        self.full_left_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_right_data = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.left_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.right_depth = np.asarray(np.random.uniform(0, 255, (32, 32)), dtype='uint8')
        self.left_labels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.right_labels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.left_normals = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.right_normals = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.full_image = im.StereoImage(
            left_data=self.full_left_data,
            right_data=self.full_right_data,
            left_camera_pose=self.full_left_pose,
            right_camera_pose=self.full_right_pose,
            left_depth_data=self.left_depth,
            right_depth_data=self.right_depth,
            left_labels_data=self.left_labels,
            right_labels_data=self.right_labels,
            left_world_normals_data=self.left_normals,
            right_world_normals_data=self.right_normals,
            metadata=metadata,
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        )

    def test_filename(self):
        self.assertNPEqual(self.image.left_data, self.left_data)
        self.assertNPEqual(self.image.right_data, self.right_data)
        self.assertNPEqual(self.full_image.left_data, self.full_left_data)
        self.assertNPEqual(self.full_image.right_data, self.full_right_data)

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

    def test_depth_filename(self):
        self.assertEqual(self.image.left_depth_data, None)
        self.assertEqual(self.image.right_depth_data, None)
        self.assertNPEqual(self.full_image.left_depth_data, self.left_depth)
        self.assertNPEqual(self.full_image.right_depth_data, self.right_depth)

    def test_labels_filename(self):
        self.assertEqual(self.image.left_labels_data, None)
        self.assertEqual(self.image.right_labels_data, None)
        self.assertNPEqual(self.full_image.left_labels_data, self.left_labels)
        self.assertNPEqual(self.full_image.right_labels_data, self.right_labels)

    def test_world_normals_filename(self):
        self.assertEqual(self.image.left_world_normals_data, None)
        self.assertEqual(self.image.right_world_normals_data, None)
        self.assertNPEqual(self.full_image.left_world_normals_data, self.left_normals)
        self.assertNPEqual(self.full_image.right_world_normals_data, self.right_normals)

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

    def test_padded_kwargs(self):
        kwargs = {
            'left_filename': '/home/user/left.png',
            'left_camera_location': np.array([1, 2, 3]),
            'left_camera_orientation': np.array([4, 5, 6, 7]),
            'right_filename': '/home/user/right.png',
            'right_camera_location': np.array([8, 9, 10]),
            'right_camera_orientation': np.array([11, 12, 13, 14]),
            # Extras:
            'a': 1, 'b': 2, 'c': 3
        }
        with self.assertRaises(TypeError):
            im.StereoImage(**kwargs)

    def test_left_image_is_base(self):
        self.assertNPEqual(self.image.data, self.image.left_data)
        self.assertNPEqual(self.image.camera_location, self.image.left_camera_location)
        self.assertNPEqual(self.image.camera_orientation, self.image.left_camera_orientation)
        self.assertNPEqual(self.image.data, self.image.left_data)
        self.assertNPEqual(self.image.depth_data, self.image.left_depth_data)
        self.assertNPEqual(self.image.labels_data, self.image.left_labels_data)
        self.assertNPEqual(self.image.world_normals_data, self.image.left_world_normals_data)

        self.assertNPEqual(self.full_image.data, self.full_image.left_data)
        self.assertNPEqual(self.full_image.camera_location, self.full_image.left_camera_location)
        self.assertNPEqual(self.full_image.camera_orientation, self.full_image.left_camera_orientation)
        self.assertNPEqual(self.full_image.data, self.full_image.left_data)
        self.assertNPEqual(self.full_image.depth_data, self.full_image.left_depth_data)
        self.assertNPEqual(self.full_image.labels_data, self.full_image.left_labels_data)
        self.assertNPEqual(self.full_image.world_normals_data, self.full_image.left_world_normals_data)

    def test_make_from_images(self):
        metadata = imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT,
            time_of_day=imeta.TimeOfDay.DAY,
            height=600,
            width=800,
            fov=90,
            focal_length=5,
            aperture=22,
            simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT,
            texture_mipmap_bias=1,
            normal_mipmap_bias=2,
            roughness_enabled=True,
            geometry_decimation=0.8,
            procedural_generation_seed=16234,
            labelled_objects=(
                imeta.LabelledObject(class_names=('car',), bounding_box=(12, 144, 67, 43), label_color=(123, 127, 112),
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)), object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',), bounding_box=(125, 244, 117, 67), label_color=(27, 89, 62),
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     object_id='cat-090')
            ),
            average_scene_depth=90.12)

        left_pose = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        left_image = im.Image(
            data=self.left_data,
            camera_pose=left_pose,
            depth_data=self.left_depth,
            labels_data=self.left_labels,
            world_normals_data=self.left_normals,
            metadata=metadata,
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        )

        right_pose = tf.Transform(location=left_pose.find_independent((0, 0, 15)),
                                  rotation=left_pose.rotation_quat(w_first=False),
                                  w_first=False)
        right_image = im.Image(
            data=self.right_data,
            camera_pose=right_pose,
            depth_data=self.right_depth,
            labels_data=self.right_labels,
            world_normals_data=self.right_normals,
            metadata=metadata,
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
        self.assertEqual(stereo_image.additional_metadata,
                         du.defaults(left_image.additional_metadata, right_image.additional_metadata))
        self.assertNPEqual(stereo_image.left_camera_location, left_image.camera_location)
        self.assertNPEqual(stereo_image.left_camera_orientation, left_image.camera_orientation)
        self.assertNPEqual(stereo_image.left_data, left_image.data)
        self.assertNPEqual(stereo_image.left_depth_data, left_image.depth_data)
        self.assertNPEqual(stereo_image.left_labels_data, left_image.labels_data)
        self.assertNPEqual(stereo_image.left_world_normals_data, left_image.world_normals_data)

        self.assertNPEqual(stereo_image.right_camera_location, right_image.camera_location)
        self.assertNPEqual(stereo_image.right_camera_orientation, right_image.camera_orientation)
        self.assertNPEqual(stereo_image.right_data, right_image.data)
        self.assertNPEqual(stereo_image.right_depth_data, right_image.depth_data)
        self.assertNPEqual(stereo_image.right_labels_data, right_image.labels_data)
        self.assertNPEqual(stereo_image.right_world_normals_data, right_image.world_normals_data)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))
