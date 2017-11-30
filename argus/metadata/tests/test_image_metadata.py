# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import bson
import argus.util.transform as tf
import argus.util.dict_utils as du
import argus.metadata.camera_intrinsics as cam_intr
import argus.metadata.image_metadata as imeta
import argus.core.image


class TestLabelledObject(unittest.TestCase):

    def test_equals(self):
        kwargs = {
            'class_names': ('class_1',),
            'bounding_box': (152, 239, 14, 78),
            'label_color': (127, 33, 67),
            'relative_pose': tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'object_id': 'LabelledObject-18569'
        }
        a = imeta.LabelledObject(**kwargs)
        b = imeta.LabelledObject(**kwargs)
        self.assertEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'class_names': ('class_41',)}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'bounding_box': (47, 123, 45, 121)}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'label_color': (247, 123, 14)}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'relative_pose': tf.Transform((62, -81, 43), (0.1, 0.1, 0.1))}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'object_id': 'Cat-12'}, kwargs))
        self.assertNotEqual(a, b)

    def test_hash(self):
        kwargs = {
            'class_names': ('class_1',),
            'bounding_box': (152, 239, 14, 78),
            'label_color': (127, 33, 67),
            'relative_pose': tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'object_id': 'LabelledObject-18569'
        }
        a = imeta.LabelledObject(**kwargs)
        b = imeta.LabelledObject(**kwargs)
        self.assertEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'class_names': 'class_41'}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'bounding_box': (47, 123, 45, 121)}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'label_color': (247, 123, 14)}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'relative_pose': tf.Transform((62, -81, 43), (0.1, 0.1, 0.1))}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'object_id': 'Cat-12'}, kwargs))
        self.assertNotEqual(hash(a), hash(b))

    def test_set(self):
        a = imeta.LabelledObject(
            class_names=('class_1',),
            bounding_box=(152, 239, 14, 78),
            label_color=(127, 33, 67),
            relative_pose=tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            object_id='LabelledObject-18569'
        )
        b = imeta.LabelledObject(
            class_names=('class_2',),
            bounding_box=(39, 169, 96, 16),
            label_color=(2, 227, 34),
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
            object_id='LabelledObject-68478'
        )
        c = imeta.LabelledObject(
            class_names=('class_3',),
            bounding_box=(148, 468, 82, 241),
            label_color=(12, 82, 238),
            relative_pose=tf.Transform(location=(85, -648, -376), rotation=(0.8, -0.64, -0.73)),
            object_id='LabelledObject-87684'
        )
        subject_set = {a, a, a, b}
        self.assertEqual(2, len(subject_set))
        self.assertIn(a, subject_set)
        self.assertIn(b, subject_set)
        self.assertNotIn(c, subject_set)

    def test_serialize_and_deserialize(self):
        obj1 = imeta.LabelledObject(
            class_names=('class_3',),
            bounding_box=(148, 468, 82, 241),
            label_color=(12, 82, 238),
            relative_pose=tf.Transform(location=(85, -648, -376), rotation=(0.8, -0.64, -0.73)),
            object_id='LabelledObject-87684'
        )
        s_obj1 = obj1.serialize()

        obj2 = imeta.LabelledObject.deserialize(s_obj1)
        s_obj2 = obj2.serialize()

        self.assertEqual(obj1, obj2)
        self.assertEqual(s_obj1, s_obj2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            obj2 = imeta.LabelledObject.deserialize(s_obj2)
            s_obj2 = obj2.serialize()
            self.assertEqual(obj1, obj2)
            self.assertEqual(s_obj1, s_obj2)


class TestImageMetadata(unittest.TestCase):

    parent_image = argus.core.image.Image(
        data=np.random.randint(0, 255, (32, 32, 3), dtype='uint8'),
        metadata=imeta.ImageMetadata(source_type=imeta.ImageSourceType.SYNTHETIC, hash_=b'\x1f`\xa8\x8aR\xed\x9f\x0b'))

    def make_metadata(self, **kwargs):
        kwargs = du.defaults(kwargs, {
            'hash_': b'\xa5\xc9\x08\xaf$\x0b\x116',
            'source_type': imeta.ImageSourceType.SYNTHETIC,
            'environment_type': imeta.EnvironmentType.INDOOR_CLOSE,
            'light_level': imeta.LightingLevel.WELL_LIT,
            'time_of_day': imeta.TimeOfDay.DAY,

            'camera_pose': tf.Transform((1, 3, 4), (0.2, 0.8, 0.2, -0.7)),
            'right_camera_pose': tf.Transform((-10, -20, -30), (0.9, -0.7, 0.5, -0.3)),
            'intrinsics': cam_intr.CameraIntrinsics(700, 700, 654.2, 753.3, 400, 300),
            'right_intrinsics': cam_intr.CameraIntrinsics(700, 710, 732.1, 612.3, 400, 300),
            'lens_focal_distance': 5,
            'aperture': 22,

            'simulator': bson.ObjectId('5a14cf0e36ed1e17a55f1e35'),
            'simulation_world': 'TestSimulationWorld',
            'lighting_model': imeta.LightingModel.LIT,
            'texture_mipmap_bias': 1,
            'normal_maps_enabled': True,
            'roughness_enabled': True,
            'geometry_decimation': 0.8,

            'procedural_generation_seed': 16234,

            'labelled_objects': [
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ],
            'average_scene_depth': 90.12,
            'base_image': self.parent_image,
            'transformation_matrix': np.array([[0.19882871, 0.58747441, 0.90084303],
                                               [0.6955363, 0.48193339, 0.09503605],
                                               [0.20549805, 0.6110534, 0.61145574]])
        })
        return imeta.ImageMetadata(**kwargs)

    def test_constructor_works_with_minimal_parameters(self):
        imeta.ImageMetadata(source_type=imeta.ImageSourceType.SYNTHETIC, hash_=b'\x1f`\xa8\x8aR\xed\x9f\x0b')

    def test_serialize_and_deserialise(self):
        entity1 = self.make_metadata()
        s_entity1 = entity1.serialize()

        entity2 = imeta.ImageMetadata.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assert_metadata_equal(entity1, entity2)
        self.assertEqual(s_entity1, s_entity2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = imeta.ImageMetadata.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assert_metadata_equal(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

    def test_serialize_and_deserialize_works_with_minimal_parameters(self):
        entity1 = imeta.ImageMetadata(source_type=imeta.ImageSourceType.SYNTHETIC, hash_=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                                      intrinsics=cam_intr.CameraIntrinsics(800, 600, 652.2, 291, 142.2, 614.4))
        s_entity1 = entity1.serialize()

        entity2 = imeta.ImageMetadata.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assert_metadata_equal(entity1, entity2)
        self.assertEqual(s_entity1, s_entity2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = imeta.ImageMetadata.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assert_metadata_equal(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

    def test_equals(self):
        alt_metadata = {
            'hash_': [b'\x1f`\xa8\x8aR\xed\x9f\x0b'],
            'source_type': [imeta.ImageSourceType.REAL_WORLD],
            'environment_type': [imeta.EnvironmentType.INDOOR, imeta.EnvironmentType.OUTDOOR_URBAN,
                                 imeta.EnvironmentType.OUTDOOR_LANDSCAPE],
            'light_level': [imeta.LightingLevel.PITCH_BLACK, imeta.LightingLevel.DIM, imeta.LightingLevel.EVENLY_LIT,
                            imeta.LightingLevel.BRIGHT],
            'time_of_day': [imeta.TimeOfDay.DAWN, imeta.TimeOfDay.MORNING, imeta.TimeOfDay.AFTERNOON,
                            imeta.TimeOfDay.TWILIGHT, imeta.TimeOfDay.NIGHT],
            'camera_pose': [tf.Transform((12, 13, 14), (-0.5, 0.3, 0.8, -0.9))],
            'right_camera_pose': [tf.Transform((11, 15, 19), (-0.2, 0.4, 0.6, -0.8))],
            'intrinsics': [cam_intr.CameraIntrinsics(900, 910, 124.8, 285.7, 640, 360)],
            'right_intrinsics': [cam_intr.CameraIntrinsics(900, 890, 257.9, 670.12, 640, 360)],
            'lens_focal_distance': [22],
            'aperture': [1.2],
            'simulator': [bson.ObjectId()],
            'simulation_world': ['TestSimulationWorld2'],
            'lighting_model': [imeta.LightingModel.UNLIT],
            'texture_mipmap_bias': [2],
            'normal_maps_enabled': [False],
            'roughness_enabled': [False],
            'geometry_decimation': [0.3],
            'procedural_generation_seed': [7329],
            'average_scene_depth': [102.33],
            'base_image': [mock.create_autospec(argus.core.image.Image)],
            'transformation_matrix': [np.random.uniform(0, 1, (3, 3))],
            'labelled_objects': [tuple(), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('cat',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 12, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 255),
                    relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            )]
        }
        a = self.make_metadata()
        b = self.make_metadata()
        self.assertEqual(a, b)

        # Change single keys, and make sure it is no longer equal
        for key, values in alt_metadata.items():
            for val in values:
                b = self.make_metadata(**{key: val})
                self.assertNotEqual(a, b, "Changing key {0} to {1} did not change equality".format(key, str(val)))

    def test_hash(self):
        alt_metadata = {
            'hash_': [b'\x1f`\xa8\x8aR\xed\x9f\x0b'],
            'source_type': [imeta.ImageSourceType.REAL_WORLD],
            'environment_type': [imeta.EnvironmentType.INDOOR, imeta.EnvironmentType.OUTDOOR_URBAN,
                                 imeta.EnvironmentType.OUTDOOR_LANDSCAPE],
            'light_level': [imeta.LightingLevel.PITCH_BLACK, imeta.LightingLevel.DIM, imeta.LightingLevel.EVENLY_LIT,
                            imeta.LightingLevel.BRIGHT],
            'time_of_day': [imeta.TimeOfDay.DAWN, imeta.TimeOfDay.MORNING, imeta.TimeOfDay.AFTERNOON,
                            imeta.TimeOfDay.TWILIGHT, imeta.TimeOfDay.NIGHT],
            'camera_pose': [tf.Transform((12, 13, 14), (-0.5, 0.3, 0.8, -0.9))],
            'right_camera_pose': [tf.Transform((11, 15, 19), (-0.2, 0.4, 0.6, -0.8))],
            'intrinsics': [cam_intr.CameraIntrinsics(900, 910, 184.9, 892.5, 640, 360)],
            'right_intrinsics': [cam_intr.CameraIntrinsics(900, 890, 963.1, 816.2, 640, 360)],
            'lens_focal_distance': [22],
            'aperture': [1.2],
            'simulator': [bson.ObjectId()],
            'simulation_world': ['TestSimulationWorld2'],
            'lighting_model': [imeta.LightingModel.UNLIT],
            'texture_mipmap_bias': [2],
            'normal_maps_enabled': [False],
            'roughness_enabled': [False],
            'geometry_decimation': [0.3],
            'procedural_generation_seed': [7329],
            'average_scene_depth': [102.33],
            'base_image': [mock.create_autospec(argus.core.image.Image)],
            'transformation_matrix': [np.random.uniform(0, 1, (3, 3))],
            'labelled_objects': [tuple(), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('cat',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 12, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165),
                                               rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 255),
                    relative_pose=tf.Transform(location=(286, -465, -165),
                                               rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            )]
        }
        a = self.make_metadata()
        b = self.make_metadata()
        self.assertEqual(hash(a), hash(b))

        # Change single keys, and make sure it is no longer equal
        for key, values in alt_metadata.items():
            for val in values:
                b = self.make_metadata(**{key: val})
                self.assertNotEqual(hash(a), hash(b),
                                    "Changing key {0} to {1} did not change the hash".format(key, str(val)))

    def test_clone(self):
        alt_metadata = {
            'hash_': [b'\x1f`\xa8\x8aR\xed\x9f\x0b'],
            'source_type': [imeta.ImageSourceType.REAL_WORLD],
            'environment_type': [imeta.EnvironmentType.INDOOR, imeta.EnvironmentType.OUTDOOR_URBAN,
                                 imeta.EnvironmentType.OUTDOOR_LANDSCAPE],
            'light_level': [imeta.LightingLevel.PITCH_BLACK, imeta.LightingLevel.DIM, imeta.LightingLevel.EVENLY_LIT,
                            imeta.LightingLevel.BRIGHT],
            'time_of_day': [imeta.TimeOfDay.DAWN, imeta.TimeOfDay.MORNING, imeta.TimeOfDay.AFTERNOON,
                            imeta.TimeOfDay.TWILIGHT, imeta.TimeOfDay.NIGHT],
            'camera_pose': [tf.Transform((12, 13, 14), (-0.5, 0.3, 0.8, -0.9))],
            'right_camera_pose': [tf.Transform((11, 15, 19), (-0.2, 0.4, 0.6, -0.8))],
            'intrinsics': [cam_intr.CameraIntrinsics(900, 910, 894.7, 861.2, 640, 360)],
            'right_intrinsics': [cam_intr.CameraIntrinsics(900, 890, 760.45, 405.1, 640, 360)],
            'lens_focal_distance': [22],
            'aperture': [1.2],
            'simulator': [bson.ObjectId()],
            'simulation_world': ['TestSimulationWorld2'],
            'lighting_model': [imeta.LightingModel.UNLIT],
            'texture_mipmap_bias': [2],
            'normal_maps_enabled': [False],
            'roughness_enabled': [False],
            'geometry_decimation': [0.3],
            'procedural_generation_seed': [7329],
            'average_scene_depth': [102.33],
            'base_image': [mock.create_autospec(argus.core.image.Image)],
            'transformation_matrix': [np.random.uniform(0, 1, (3, 3))],
            'labelled_objects': [tuple(), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('cat',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 12, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 163),
                    relative_pose=tf.Transform(location=(286, -465, -165),
                                               rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            ), (
                imeta.LabelledObject(
                    class_names=('cup',),
                    bounding_box=(142, 280, 54, 78),
                    label_color=(2, 227, 34),
                    relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                    object_id='LabelledObject-68478'
                ),
                imeta.LabelledObject(
                    class_names=('car',),
                    bounding_box=(542, 83, 63, 123),
                    label_color=(26, 12, 212),
                    relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                    object_id='LabelledObject-8246'
                ),
                imeta.LabelledObject(
                    class_names=('cow',),
                    bounding_box=(349, 672, 124, 208),
                    label_color=(162, 134, 255),
                    relative_pose=tf.Transform(location=(286, -465, -165),
                                               rotation=(0.9, 0.1, 0.5)),
                    object_id='LabelledObject-56485'
                )
            )]
        }
        a = self.make_metadata()
        b = a.clone()
        self.assert_metadata_equal(a, b)

        # Change single keys, and make sure it is no longer equal
        for key, values in alt_metadata.items():
            for val in values:
                b = a.clone(**{key: val})

                if key == 'hash_':
                    self.assertEqual(val, b.hash)
                    self.assertNotEqual(a.hash, b.hash)
                else:
                    self.assertEqual(a.hash, b.hash)
                if key == 'source_type':
                    self.assertEqual(val, b.source_type)
                    self.assertNotEqual(a.source_type, b.source_type)
                else:
                    self.assertEqual(a.source_type, b.source_type)
                if key == 'environment_type':
                    self.assertEqual(val, b.environment_type)
                    self.assertNotEqual(a.environment_type, b.environment_type)
                else:
                    self.assertEqual(a.environment_type, b.environment_type)
                if key == 'light_level':
                    self.assertEqual(val, b.light_level)
                    self.assertNotEqual(a.light_level, b.light_level)
                else:
                    self.assertEqual(a.light_level, b.light_level)
                if key == 'time_of_day':
                    self.assertEqual(val, b.time_of_day)
                    self.assertNotEqual(a.time_of_day, b.time_of_day)
                else:
                    self.assertEqual(a.time_of_day, b.time_of_day)
                if key == 'camera_pose':
                    self.assertEqual(val, b.camera_pose)
                    self.assertNotEqual(a.camera_pose, b.camera_pose)
                else:
                    self.assertEqual(a.camera_pose, b.camera_pose)
                if key == 'right_camera_pose':
                    self.assertEqual(val, b.right_camera_pose)
                    self.assertNotEqual(a.right_camera_pose, b.right_camera_pose)
                else:
                    self.assertEqual(a.right_camera_pose, b.right_camera_pose)
                if key == 'intrinsics':
                    self.assertEqual(val, b.camera_intrinsics)
                    self.assertNotEqual(a.camera_intrinsics, b.camera_intrinsics)
                else:
                    self.assertEqual(a.camera_intrinsics, b.camera_intrinsics)
                    self.assertEqual(a.width, b.width)
                    self.assertEqual(a.height, b.height)
                if key == 'right_intrinsics':
                    self.assertEqual(val, b.right_camera_intrinsics)
                    self.assertNotEqual(a.right_camera_intrinsics, b.right_camera_intrinsics)
                else:
                    self.assertEqual(a.right_camera_intrinsics, b.right_camera_intrinsics)
                if key == 'lens_focal_distance':
                    self.assertEqual(val, b.lens_focal_distance)
                    self.assertNotEqual(a.lens_focal_distance, b.lens_focal_distance)
                else:
                    self.assertEqual(a.lens_focal_distance, b.lens_focal_distance)
                if key == 'aperture':
                    self.assertEqual(val, b.aperture)
                    self.assertNotEqual(a.aperture, b.aperture)
                else:
                    self.assertEqual(a.aperture, b.aperture)
                if key == 'simulation_world':
                    self.assertEqual(val, b.simulation_world)
                    self.assertNotEqual(a.simulation_world, b.simulation_world)
                else:
                    self.assertEqual(a.simulation_world, b.simulation_world)
                if key == 'lighting_model':
                    self.assertEqual(val, b.lighting_model)
                    self.assertNotEqual(a.lighting_model, b.lighting_model)
                else:
                    self.assertEqual(a.lighting_model, b.lighting_model)
                if key == 'texture_mipmap_bias':
                    self.assertEqual(val, b.texture_mipmap_bias)
                    self.assertNotEqual(a.texture_mipmap_bias, b.texture_mipmap_bias)
                else:
                    self.assertEqual(a.texture_mipmap_bias, b.texture_mipmap_bias)
                if key == 'normal_maps_enabled':
                    self.assertEqual(val, b.normal_maps_enabled)
                    self.assertNotEqual(a.normal_maps_enabled, b.normal_maps_enabled)
                else:
                    self.assertEqual(a.normal_maps_enabled, b.normal_maps_enabled)
                if key == 'roughness_enabled':
                    self.assertEqual(val, b.roughness_enabled)
                    self.assertNotEqual(a.roughness_enabled, b.roughness_enabled)
                else:
                    self.assertEqual(a.roughness_enabled, b.roughness_enabled)
                if key == 'geometry_decimation':
                    self.assertEqual(val, b.geometry_decimation)
                    self.assertNotEqual(a.geometry_decimation, b.geometry_decimation)
                else:
                    self.assertEqual(a.geometry_decimation, b.geometry_decimation)
                if key == 'procedural_generation_seed':
                    self.assertEqual(val, b.procedural_generation_seed)
                    self.assertNotEqual(a.procedural_generation_seed, b.procedural_generation_seed)
                else:
                    self.assertEqual(a.procedural_generation_seed, b.procedural_generation_seed)
                if key == 'labelled_objects':
                    self.assertEqual(val, b.labelled_objects)
                    self.assertNotEqual(a.labelled_objects, b.labelled_objects)
                else:
                    self.assertEqual(a.labelled_objects, b.labelled_objects)
                if key == 'average_scene_depth':
                    self.assertEqual(val, b.average_scene_depth)
                    self.assertNotEqual(a.average_scene_depth, b.average_scene_depth)
                else:
                    self.assertEqual(a.average_scene_depth, b.average_scene_depth)

    def assert_metadata_equal(self, metadata1, metadata2):
        if not isinstance(metadata1, imeta.ImageMetadata):
            self.fail("metadata 1 is not an image metadata")
        if not isinstance(metadata2, imeta.ImageMetadata):
            self.fail("metadata 1 is not an image metadata")
        self.assertEqual(metadata1.hash, metadata2.hash)
        self.assertEqual(metadata1.source_type, metadata2.source_type)
        self.assertEqual(metadata1.environment_type, metadata2.environment_type)
        self.assertEqual(metadata1.light_level, metadata2.light_level)
        self.assertEqual(metadata1.time_of_day, metadata2.time_of_day)
        self.assertEqual(metadata1.height, metadata2.height)
        self.assertEqual(metadata1.width, metadata2.width)
        self.assertEqual(metadata1.camera_pose, metadata2.camera_pose)
        self.assertEqual(metadata1.right_camera_pose, metadata2.right_camera_pose)
        self.assertEqual(metadata1.camera_intrinsics, metadata2.camera_intrinsics)
        self.assertEqual(metadata1.right_camera_intrinsics, metadata2.right_camera_intrinsics)
        self.assertEqual(metadata1.lens_focal_distance, metadata2.lens_focal_distance)
        self.assertEqual(metadata1.aperture, metadata2.aperture)
        self.assertEqual(metadata1.simulator, metadata2.simulator)
        self.assertEqual(metadata1.simulation_world, metadata2.simulation_world)
        self.assertEqual(metadata1.lighting_model, metadata2.lighting_model)
        self.assertEqual(metadata1.texture_mipmap_bias, metadata2.texture_mipmap_bias)
        self.assertEqual(metadata1.normal_maps_enabled, metadata2.normal_maps_enabled)
        self.assertEqual(metadata1.roughness_enabled, metadata2.roughness_enabled)
        self.assertEqual(metadata1.geometry_decimation, metadata2.geometry_decimation)
        self.assertEqual(metadata1.procedural_generation_seed, metadata2.procedural_generation_seed)
        self.assertEqual(metadata1.labelled_objects, metadata2.labelled_objects)
        self.assertEqual(metadata1.average_scene_depth, metadata2.average_scene_depth)
        self.assertEqual(metadata1.base_image, metadata2.base_image)
        self.assertTrue(np.array_equal(metadata1.affine_transformation_matrix, metadata2.affine_transformation_matrix))
