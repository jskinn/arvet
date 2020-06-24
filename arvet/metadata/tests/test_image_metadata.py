# Copyright (c) 2017, John Skinner
import os.path
import unittest
import numpy as np
import pymodm
import xxhash
import arvet.util.transform as tf
import arvet.util.dict_utils as du
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta


# -------------------- LABELLED OBJECT --------------------

class TestLabelledObject(unittest.TestCase):

    def test_equals(self):
        kwargs = {
            'class_names': ('class_1',),
            'x': 152,
            'y': 239,
            'width': 15,
            'height': 25,
            'relative_pose': tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'instance_name': 'LabelledObject-18569'
        }
        a = imeta.LabelledObject(**kwargs)
        b = imeta.LabelledObject(**kwargs)
        self.assertEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'class_names': ('class_41',)}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'x': 47}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'y': 123}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'width': 34}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'height': 34}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({
            'relative_pose': tf.Transform((62, -81, 43), (0.1, 0.1, 0.1))}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.LabelledObject(**du.defaults({'instance_name': 'Cat-12'}, kwargs))
        self.assertNotEqual(a, b)

    def test_hash(self):
        kwargs = {
            'class_names': ('class_1',),
            'x': 152,
            'y': 239,
            'width': 15,
            'height': 25,
            'relative_pose': tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'instance_name': 'LabelledObject-18569'
        }
        a = imeta.LabelledObject(**kwargs)
        b = imeta.LabelledObject(**kwargs)
        self.assertEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'class_names': 'class_41'}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'x': 47}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'y': 123}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'width': 34}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'height': 34}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({
            'relative_pose': tf.Transform((62, -81, 43), (0.1, 0.1, 0.1))}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.LabelledObject(**du.defaults({'instance_name': 'Cat-12'}, kwargs))
        self.assertNotEqual(hash(a), hash(b))

    def test_set(self):
        a = imeta.LabelledObject(
            class_names=('class_1',),
            x=152,
            y=239,
            width=15,
            height=25,
            relative_pose=tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            instance_name='LabelledObject-18569'
        )
        b = imeta.LabelledObject(
            class_names=('class_2',),
            x=39,
            y=169,
            width=45,
            height=75,
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
            instance_name='LabelledObject-68478'
        )
        c = imeta.LabelledObject(
            class_names=('class_3',),
            x=148,
            y=468,
            width=36,
            height=85,
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.8, -0.64, -0.73)),
            instance_name='LabelledObject-87684'
        )
        subject_set = {a, a, a, b}
        self.assertEqual(2, len(subject_set))
        self.assertIn(a, subject_set)
        self.assertIn(b, subject_set)
        self.assertNotIn(c, subject_set)

    def test_to_and_from_son(self):
        obj1 = imeta.LabelledObject(
            class_names=('class_3',),
            x=148,
            y=468,
            width=15,
            height=25,
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.8, -0.64, -0.73)),
            instance_name='LabelledObject-87684'
        )
        s_obj1 = obj1.to_son()

        obj2 = imeta.LabelledObject.from_document(s_obj1)
        s_obj2 = obj2.to_son()

        self.assertEqual(obj1, obj2)
        self.assertEqual(s_obj1, s_obj2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            obj2 = imeta.LabelledObject.from_document(s_obj2)
            s_obj2 = obj2.to_son()
            self.assertEqual(obj1, obj2)
            self.assertEqual(s_obj1, s_obj2)


class TestLabelledObjectMongoModel(pymodm.MongoModel):
    object = pymodm.fields.EmbeddedDocumentField(imeta.LabelledObject)


class TestLabelledObjectDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        TestLabelledObjectMongoModel._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        TestLabelledObjectMongoModel._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = imeta.LabelledObject(
            class_names=('class_1',),
            x=152,
            y=239,
            width=15,
            height=25,
            relative_pose=tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            instance_name='LabelledObject-18569'
        )

        # Save the model
        model = TestLabelledObjectMongoModel()
        model.object = obj
        model.save()

        # Load all the entities
        all_entities = list(TestLabelledObjectMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].object, obj)
        all_entities[0].delete()

    def test_stores_and_loads_minimal(self):
        obj = imeta.LabelledObject(
            class_names=('class_1',),
            x=152,
            y=239,
            width=15,
            height=25
        )

        # Save the model
        model = TestLabelledObjectMongoModel()
        model.object = obj
        model.save()

        # Load all the entities
        all_entities = list(TestLabelledObjectMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].object, obj)
        all_entities[0].delete()


# -------------------- MASKED OBJECT --------------------

class TestMaskedObject(unittest.TestCase):

    def test_incorrect_size(self):
        with self.assertRaises(ValueError):
            imeta.MaskedObject(
                ('class_1',),
                152,
                239,
                140,
                78,
                tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
                'LabelledObject-18569',
                np.random.choice((True, False), size=(78, 14))
            )
        with self.assertRaises(ValueError):
            imeta.MaskedObject(
                ('class_1',),
                152,
                239,
                14,
                87,
                tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
                'LabelledObject-18569',
                np.random.choice((True, False), size=(78, 14))
            )
        with self.assertRaises(ValueError):
            imeta.MaskedObject(
                class_names=('class_1',),
                x=152,
                y=239,
                width=140,
                height=78,
                mask=np.random.choice((True, False), size=(78, 14))
            )
        with self.assertRaises(ValueError):
            imeta.MaskedObject(
                class_names=('class_1',),
                x=152,
                y=239,
                width=14,
                height=87,
                mask=np.random.choice((True, False), size=(78, 14))
            )

    def test_reads_size_from_mask_arg(self):
        obj = imeta.MaskedObject(
            ('class_1',),
            152,
            239,
            78,
            14,
            tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'LabelledObject-18569',
            np.random.choice((True, False), size=(14, 78))
        )
        self.assertEqual(obj.width, 78)
        self.assertEqual(obj.height, 14)

    def test_reads_size_from_mask_kwarg(self):
        obj = imeta.MaskedObject(
            class_names=('class_1',),
            x=152,
            y=239,
            mask=np.random.choice((True, False), size=(14, 78))
        )
        self.assertEqual(obj.width, 78)
        self.assertEqual(obj.height, 14)

    def test_equals(self):
        kwargs = {
            'class_names': ('class_1',),
            'x': 152,
            'y': 239,
            'mask': np.random.choice((True, False), size=(14, 78)),
            'relative_pose': tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'instance_name': 'LabelledObject-18569'
        }
        a = imeta.MaskedObject(**kwargs)
        b = imeta.MaskedObject(**kwargs)
        self.assertEqual(a, b)
        b = imeta.MaskedObject(**du.defaults({'class_names': ('class_41',)}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.MaskedObject(**du.defaults({'x': 47}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.MaskedObject(**du.defaults({'y': 123}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.MaskedObject(**du.defaults({'mask': np.random.choice((True, False), size=(14, 78))}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.MaskedObject(**du.defaults({
            'relative_pose': tf.Transform((62, -81, 43), (0.1, 0.1, 0.1))}, kwargs))
        self.assertNotEqual(a, b)
        b = imeta.MaskedObject(**du.defaults({'instance_name': 'Cat-12'}, kwargs))
        self.assertNotEqual(a, b)

    def test_hash(self):
        kwargs = {
            'class_names': ('class_1',),
            'x': 152,
            'y': 239,
            'mask': np.random.choice((True, False), size=(14, 78)),
            'relative_pose': tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            'instance_name': 'LabelledObject-18569'
        }
        a = imeta.MaskedObject(**kwargs)
        b = imeta.MaskedObject(**kwargs)
        self.assertEqual(hash(a), hash(b))
        b = imeta.MaskedObject(**du.defaults({'class_names': 'class_41'}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.MaskedObject(**du.defaults({'x': 47}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.MaskedObject(**du.defaults({'y': 123}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.MaskedObject(**du.defaults({'mask': np.random.choice((True, False), size=(14, 78))}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.MaskedObject(**du.defaults({
            'relative_pose': tf.Transform((62, -81, 43), (0.1, 0.1, 0.1))}, kwargs))
        self.assertNotEqual(hash(a), hash(b))
        b = imeta.MaskedObject(**du.defaults({'instance_name': 'Cat-12'}, kwargs))
        self.assertNotEqual(hash(a), hash(b))

    def test_set(self):
        a = imeta.MaskedObject(
            class_names=('class_1',),
            x=152,
            y=239,
            mask=np.random.choice((True, False), size=(14, 78)),
            relative_pose=tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            instance_name='LabelledObject-18569'
        )
        b = imeta.MaskedObject(
            class_names=('class_2',),
            x=39,
            y=169,
            mask=np.random.choice((True, False), size=(14, 78)),
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
            instance_name='LabelledObject-68478'
        )
        c = imeta.MaskedObject(
            class_names=('class_3',),
            x=148,
            y=468,
            mask=np.random.choice((True, False), size=(14, 78)),
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.8, -0.64, -0.73)),
            instance_name='LabelledObject-87684'
        )
        subject_set = {a, a, a, b}
        self.assertEqual(2, len(subject_set))
        self.assertIn(a, subject_set)
        self.assertIn(b, subject_set)
        self.assertNotIn(c, subject_set)

    def test_to_and_from_son(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        obj1 = imeta.MaskedObject(
            class_names=('class_3',),
            x=148,
            y=468,
            mask=np.random.choice((True, False), size=(14, 78)),
            relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.8, -0.64, -0.73)),
            instance_name='LabelledObject-87684'
        )
        s_obj1 = obj1.to_son()

        obj2 = imeta.MaskedObject.from_document(s_obj1)
        s_obj2 = obj2.to_son()

        self.assertEqual(obj1, obj2)
        self.assertEqual(s_obj1, s_obj2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            obj2 = imeta.MaskedObject.from_document(s_obj2)
            s_obj2 = obj2.to_son()
            self.assertEqual(obj1, obj2)
            self.assertEqual(s_obj1, s_obj2)


class TestMaskedObjectMongoModel(pymodm.MongoModel):
    object = pymodm.fields.EmbeddedDocumentField(imeta.MaskedObject)


class TestMaskedObjectDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        TestMaskedObjectMongoModel._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        TestMaskedObjectMongoModel._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = imeta.MaskedObject(
            class_names=('class_1',),
            x=152,
            y=239,
            width=14,
            height=78,
            mask=np.random.choice((True, False), size=(78, 14)),
            relative_pose=tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
            instance_name='LabelledObject-18569'
        )

        # Save the model
        model = TestMaskedObjectMongoModel()
        model.object = obj
        model.save()

        # Load all the entities
        all_entities = list(TestMaskedObjectMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].object, obj)
        all_entities[0].delete()

    def test_stores_and_loads_minimal(self):
        obj = imeta.MaskedObject(
            class_names=('class_1',),
            x=152,
            y=239,
            mask=np.random.choice((True, False), size=(14, 78)),
        )

        # Save the model
        model = TestMaskedObjectMongoModel()
        model.object = obj
        model.save()

        # Load all the entities
        all_entities = list(TestMaskedObjectMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].object, obj)
        all_entities[0].delete()


# -------------------- IMAGE METADATA --------------------

def make_metadata(**kwargs):
    kwargs = du.defaults(kwargs, {
        'img_hash': b'\xa5\xc9\x08\xaf$\x0b\x116',
        'source_type': imeta.ImageSourceType.SYNTHETIC,

        'camera_pose': tf.Transform((1, 3, 4), (0.2, 0.8, 0.2, -0.7)),
        'intrinsics': cam_intr.CameraIntrinsics(700, 700, 654.2, 753.3, 400, 300),
        'lens_focal_distance': 5,
        'aperture': 22,

        'red_mean': 144.2,
        'red_std': 5.6,
        'green_mean': 128.3,
        'green_std': 45.7,
        'blue_mean': 75,
        'blue_std': 9.2,
        'depth_mean': 227.3,
        'depth_std': 543.2,

        'environment_type': imeta.EnvironmentType.INDOOR_CLOSE,
        'light_level': imeta.LightingLevel.WELL_LIT,
        'time_of_day': imeta.TimeOfDay.DAY,

        'simulation_world': 'TestSimulationWorld',
        'lighting_model': imeta.LightingModel.LIT,
        'texture_mipmap_bias': 1,
        'normal_maps_enabled': True,
        'roughness_enabled': True,
        'geometry_decimation': 0.8,
        'minimum_object_volume': 0.236,

        'procedural_generation_seed': 16234,
        'labelled_objects': [
            imeta.LabelledObject(
                class_names=('cup',),
                x=142, y=280, width=54, height=78,
                relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                instance_name='LabelledObject-68478'
            ),
            imeta.LabelledObject(
                class_names=('car',),
                x=542, y=83, width=63, height=123,
                relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                instance_name='LabelledObject-8246'
            ),
            imeta.LabelledObject(
                class_names=('cow',),
                x=349, y=672, width=124, height=208,
                relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                instance_name='LabelledObject-56485'
            )
        ]
    })
    return imeta.ImageMetadata(**kwargs)


class TestImageMetadata(unittest.TestCase):
    alt_metadata = {
        'img_hash': [b'\x1f`\xa8\x8aR\xed\x9f\x0b'],
        'source_type': [imeta.ImageSourceType.REAL_WORLD],

        'camera_pose': [tf.Transform((12, 13, 14), (-0.5, 0.3, 0.8, -0.9))],
        'intrinsics': [cam_intr.CameraIntrinsics(900, 910, 124.8, 285.7, 640, 360)],
        'lens_focal_distance': [22],
        'aperture': [1.2],

        'red_mean': [15.2],
        'red_std': [15.2],
        'green_mean': [15.2],
        'green_std': [15.2],
        'blue_mean': [15.2],
        'blue_std': [15.2],
        'depth_mean': [15.2],
        'depth_std': [15.2],

        'environment_type': [imeta.EnvironmentType.INDOOR, imeta.EnvironmentType.OUTDOOR_URBAN,
                             imeta.EnvironmentType.OUTDOOR_LANDSCAPE],
        'light_level': [imeta.LightingLevel.PITCH_BLACK, imeta.LightingLevel.DIM, imeta.LightingLevel.EVENLY_LIT,
                        imeta.LightingLevel.BRIGHT],
        'time_of_day': [imeta.TimeOfDay.DAWN, imeta.TimeOfDay.MORNING, imeta.TimeOfDay.AFTERNOON,
                        imeta.TimeOfDay.TWILIGHT, imeta.TimeOfDay.NIGHT],

        'simulation_world': ['TestSimulationWorld2'],
        'lighting_model': [imeta.LightingModel.UNLIT],
        'texture_mipmap_bias': [2],
        'normal_maps_enabled': [False],
        'roughness_enabled': [False],
        'geometry_decimation': [0.3],
        'minimum_object_volume': [0.9887],

        'procedural_generation_seed': [7329],
        'labelled_objects': [tuple(), (
            imeta.LabelledObject(
                class_names=('cup',),
                x=142, y=280, width=54, height=78,
                relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                instance_name='LabelledObject-68478'
            ),
            imeta.LabelledObject(
                class_names=('cat',),   # Changed class
                x=542, y=83, width=63, height=123,
                relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                instance_name='LabelledObject-8246'
            ),
            imeta.LabelledObject(
                class_names=('cow',),
                x=349, y=672, width=124, height=208,
                relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                instance_name='LabelledObject-56485'
            )
        ), (
            imeta.LabelledObject(
                class_names=('cup',),
                x=142, y=280, width=54, height=78,
                relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                instance_name='LabelledObject-68478'
            ),
            imeta.LabelledObject(
                class_names=('car',),
                x=542, y=83, width=63, height=123,
                relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                instance_name='LabelledObject-8246'
            ),
            imeta.LabelledObject(
                class_names=('cow',),
                x=34, y=672, width=124, height=208,  # x changed
                relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                instance_name='LabelledObject-56485'
            )
        ), (
            imeta.LabelledObject(
                class_names=('cup',),
                x=142, y=280, width=54, height=78,
                relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                instance_name='LabelledObject-10'    # Changed
            ),
            imeta.LabelledObject(
                class_names=('car',),
                x=542, y=83, width=63, height=123,
                relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                instance_name='LabelledObject-8246'
            ),
            imeta.LabelledObject(
                class_names=('cow',),
                x=349, y=672, width=124, height=208,
                relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                instance_name='LabelledObject-56485'
            )
        )]
    }

    def test_constructor_works_with_minimal_parameters(self):
        imeta.ImageMetadata(source_type=imeta.ImageSourceType.SYNTHETIC, img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b')

    def test_to_and_from_son(self):
        entity1 = make_metadata()
        s_entity1 = entity1.to_son()

        entity2 = imeta.ImageMetadata.from_document(s_entity1)
        s_entity2 = entity2.to_son()

        self.assert_metadata_equal(entity1, entity2)
        self.assertEqual(s_entity1, s_entity2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = imeta.ImageMetadata.from_document(s_entity2)
            s_entity2 = entity2.to_son()
            self.assert_metadata_equal(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

    def test_to_and_from_son_works_with_minimal_parameters(self):
        entity1 = imeta.ImageMetadata(source_type=imeta.ImageSourceType.SYNTHETIC,
                                      img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                                      intrinsics=cam_intr.CameraIntrinsics(800, 600, 652.2, 291, 142.2, 614.4))
        s_entity1 = entity1.to_son()

        entity2 = imeta.ImageMetadata.from_document(s_entity1)
        s_entity2 = entity2.to_son()

        self.assert_metadata_equal(entity1, entity2)
        self.assertEqual(s_entity1, s_entity2)

        for idx in range(100):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = imeta.ImageMetadata.from_document(s_entity2)
            s_entity2 = entity2.to_son()
            self.assert_metadata_equal(entity1, entity2)
            self.assertEqual(s_entity1, s_entity2)

    def test_equals(self):
        a = make_metadata()
        b = make_metadata()
        self.assertEqual(a, b)

        # Change single keys, and make sure it is no longer equal
        for key, values in self.alt_metadata.items():
            for val in values:
                b = make_metadata(**{key: val})
                self.assertNotEqual(a, b, "Changing key {0} to did not change equality".format(key, str(val)))

    def test_hash(self):
        a = make_metadata()
        b = make_metadata()
        self.assertEqual(hash(a), hash(b))

        # Change single keys, and make sure it is no longer equal
        for key, values in self.alt_metadata.items():
            for val in values:
                b = make_metadata(**{key: val})
                self.assertNotEqual(hash(a), hash(b),
                                    "Changing key {0} to {1} did not change the hash".format(key, str(val)))

    def assert_metadata_equal(self, metadata1, metadata2):
        if not isinstance(metadata1, imeta.ImageMetadata):
            self.fail("metadata 1 is not an image metadata")
        if not isinstance(metadata2, imeta.ImageMetadata):
            self.fail("metadata 1 is not an image metadata")
        self.assertEqual(metadata1.img_hash, metadata2.img_hash)
        self.assertEqual(metadata1.source_type, metadata2.source_type)

        self.assertEqual(metadata1.camera_pose, metadata2.camera_pose)
        self.assertEqual(metadata1.intrinsics, metadata2.intrinsics)
        self.assertEqual(metadata1.lens_focal_distance, metadata2.lens_focal_distance)
        self.assertEqual(metadata1.aperture, metadata2.aperture)

        self.assertEqual(metadata1.red_mean, metadata2.red_mean)
        self.assertEqual(metadata1.red_std, metadata2.red_std)
        self.assertEqual(metadata1.green_mean, metadata2.green_mean)
        self.assertEqual(metadata1.green_std, metadata2.green_std)
        self.assertEqual(metadata1.blue_mean, metadata2.blue_mean)
        self.assertEqual(metadata1.blue_std, metadata2.blue_std)
        self.assertEqual(metadata1.depth_mean, metadata2.depth_mean)
        self.assertEqual(metadata1.depth_std, metadata2.depth_std)

        self.assertEqual(metadata1.environment_type, metadata2.environment_type)
        self.assertEqual(metadata1.light_level, metadata2.light_level)
        self.assertEqual(metadata1.time_of_day, metadata2.time_of_day)

        self.assertEqual(metadata1.simulation_world, metadata2.simulation_world)
        self.assertEqual(metadata1.lighting_model, metadata2.lighting_model)
        self.assertEqual(metadata1.texture_mipmap_bias, metadata2.texture_mipmap_bias)
        self.assertEqual(metadata1.normal_maps_enabled, metadata2.normal_maps_enabled)
        self.assertEqual(metadata1.roughness_enabled, metadata2.roughness_enabled)
        self.assertEqual(metadata1.minimum_object_volume, metadata2.minimum_object_volume)
        self.assertEqual(metadata1.geometry_decimation, metadata2.geometry_decimation)

        self.assertEqual(metadata1.procedural_generation_seed, metadata2.procedural_generation_seed)
        self.assertEqual(metadata1.labelled_objects, metadata2.labelled_objects)


class TestImageMetadataMongoModel(pymodm.MongoModel):
    object = pymodm.fields.EmbeddedDocumentField(imeta.ImageMetadata)


class ImageMetadataDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        TestImageMetadataMongoModel._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        TestImageMetadataMongoModel._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = make_metadata()

        # Save the model
        model = TestImageMetadataMongoModel()
        model.object = obj
        model.save()

        # Load all the entities
        all_entities = list(TestImageMetadataMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].object, obj)
        all_entities[0].delete()

    def test_stores_and_loads_minimal(self):
        obj = imeta.ImageMetadata(source_type=imeta.ImageSourceType.SYNTHETIC, img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b')

        # Save the model
        model = TestImageMetadataMongoModel()
        model.object = obj
        model.save()

        # Load all the entities
        all_entities = list(TestImageMetadataMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].object, obj)
        all_entities[0].delete()


class TestMakeMetadata(unittest.TestCase):

    def test_infers_hash_from_pixels(self):
        pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        img_hash = xxhash.xxh64(pixels).digest()
        metadata = imeta.make_metadata(pixels)
        self.assertEqual(img_hash, bytes(metadata.img_hash))

    def test_infers_image_statistics_from_pixels(self):
        pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        mean_red = np.mean(pixels[:, :, 0])
        mean_green = np.mean(pixels[:, :, 1])
        mean_blue = np.mean(pixels[:, :, 2])
        std_red = np.std(pixels[:, :, 0])
        std_green = np.std(pixels[:, :, 1])
        std_blue = np.std(pixels[:, :, 2])

        metadata = imeta.make_metadata(pixels)
        self.assertEqual(mean_red, metadata.red_mean)
        self.assertEqual(mean_green, metadata.green_mean)
        self.assertEqual(mean_blue, metadata.blue_mean)
        self.assertEqual(std_red, metadata.red_std)
        self.assertEqual(std_green, metadata.green_std)
        self.assertEqual(std_blue, metadata.blue_std)

    def test_infers_depth_statistics_from_depth(self):
        pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        depth = np.random.normal(100, 50, size=(10, 10))

        metadata = imeta.make_metadata(pixels, depth=depth)
        self.assertEqual(np.mean(depth), metadata.depth_mean)
        self.assertEqual(np.std(depth), metadata.depth_std)

    def test_overrides_arguments(self):
        pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        depth = np.random.normal(100, 50, size=(100, 100))

        true_hash = xxhash.xxh64(pixels).digest()
        mean_red = np.mean(pixels[:, :, 0])
        mean_green = np.mean(pixels[:, :, 1])
        mean_blue = np.mean(pixels[:, :, 2])
        std_red = np.std(pixels[:, :, 0])
        std_green = np.std(pixels[:, :, 1])
        std_blue = np.std(pixels[:, :, 2])

        metadata = imeta.make_metadata(
            pixels, depth,
            img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            red_mean=1000,
            blue_mean=1000,
            green_mean=1000,
            red_std=1000,
            green_std=1000,
            blue_std=1000,
            depth_mean=1000,
            depth_std=1000
        )

        self.assertEqual(true_hash, bytes(metadata.img_hash))
        self.assertEqual(mean_red, metadata.red_mean)
        self.assertEqual(mean_green, metadata.green_mean)
        self.assertEqual(mean_blue, metadata.blue_mean)
        self.assertEqual(std_red, metadata.red_std)
        self.assertEqual(std_green, metadata.green_std)
        self.assertEqual(std_blue, metadata.blue_std)
        self.assertEqual(np.mean(depth), metadata.depth_mean)
        self.assertEqual(np.std(depth), metadata.depth_std)


class TestMakeRightMetadata(unittest.TestCase):

    def test_copies_some_metadata_fields(self):
        right_pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        left_metadata = make_metadata()

        right_metadata = imeta.make_right_metadata(right_pixels, left_metadata)

        # These should be the same
        self.assertEqual(left_metadata.source_type, right_metadata.source_type)

        self.assertEqual(left_metadata.environment_type, right_metadata.environment_type)
        self.assertEqual(left_metadata.light_level, right_metadata.light_level)
        self.assertEqual(left_metadata.time_of_day, right_metadata.time_of_day)

        self.assertEqual(left_metadata.simulation_world, right_metadata.simulation_world)
        self.assertEqual(left_metadata.lighting_model, right_metadata.lighting_model)
        self.assertEqual(left_metadata.texture_mipmap_bias, right_metadata.texture_mipmap_bias)
        self.assertEqual(left_metadata.normal_maps_enabled, right_metadata.normal_maps_enabled)
        self.assertEqual(left_metadata.roughness_enabled, right_metadata.roughness_enabled)
        self.assertEqual(left_metadata.geometry_decimation, right_metadata.geometry_decimation)
        self.assertEqual(left_metadata.minimum_object_volume, right_metadata.minimum_object_volume)
        self.assertEqual(left_metadata.procedural_generation_seed, right_metadata.procedural_generation_seed)

        # These should not be
        self.assertNotEqual(left_metadata.img_hash, right_metadata.img_hash)
        self.assertNotEqual(left_metadata.camera_pose, right_metadata.camera_pose)
        self.assertNotEqual(left_metadata.intrinsics, right_metadata.intrinsics)
        self.assertNotEqual(left_metadata.lens_focal_distance, right_metadata.lens_focal_distance)
        self.assertNotEqual(left_metadata.aperture, right_metadata.aperture)

        self.assertNotEqual(left_metadata.red_mean, right_metadata.red_mean)
        self.assertNotEqual(left_metadata.red_std, right_metadata.red_std)
        self.assertNotEqual(left_metadata.green_mean, right_metadata.green_mean)
        self.assertNotEqual(left_metadata.green_std, right_metadata.green_std)
        self.assertNotEqual(left_metadata.blue_mean, right_metadata.blue_mean)
        self.assertNotEqual(left_metadata.blue_std, right_metadata.blue_std)
        self.assertNotEqual(left_metadata.depth_mean, right_metadata.depth_mean)
        self.assertNotEqual(left_metadata.depth_std, right_metadata.depth_std)

        self.assertNotEqual(left_metadata.labelled_objects, right_metadata.labelled_objects)

    def test_infers_hash_from_pixels(self):
        right_pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        left_metadata = make_metadata()
        img_hash = xxhash.xxh64(right_pixels).digest()
        right_metadata = imeta.make_right_metadata(right_pixels, left_metadata)
        self.assertEqual(img_hash, bytes(right_metadata.img_hash))

    def test_infers_image_statistics_from_pixels(self):
        pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        left_metadata = make_metadata()
        mean_red = np.mean(pixels[:, :, 0])
        mean_green = np.mean(pixels[:, :, 1])
        mean_blue = np.mean(pixels[:, :, 2])
        std_red = np.std(pixels[:, :, 0])
        std_green = np.std(pixels[:, :, 1])
        std_blue = np.std(pixels[:, :, 2])

        right_metadata = imeta.make_right_metadata(pixels, left_metadata)
        self.assertEqual(mean_red, right_metadata.red_mean)
        self.assertEqual(mean_green, right_metadata.green_mean)
        self.assertEqual(mean_blue, right_metadata.blue_mean)
        self.assertEqual(std_red, right_metadata.red_std)
        self.assertEqual(std_green, right_metadata.green_std)
        self.assertEqual(std_blue, right_metadata.blue_std)

    def test_infers_depth_statistics_from_depth(self):
        pixels = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
        depth = np.random.normal(100, 50, size=(10, 10))
        left_metadata = make_metadata()

        right_metadata = imeta.make_right_metadata(pixels, left_metadata, depth=depth)
        self.assertEqual(np.mean(depth), right_metadata.depth_mean)
        self.assertEqual(np.std(depth), right_metadata.depth_std)

    def test_overrides_arguments(self):
        pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        depth = np.random.normal(100, 50, size=(100, 100))
        left_metadata = make_metadata()

        true_hash = xxhash.xxh64(pixels).digest()
        mean_red = np.mean(pixels[:, :, 0])
        mean_green = np.mean(pixels[:, :, 1])
        mean_blue = np.mean(pixels[:, :, 2])
        std_red = np.std(pixels[:, :, 0])
        std_green = np.std(pixels[:, :, 1])
        std_blue = np.std(pixels[:, :, 2])

        right_metadata = imeta.make_right_metadata(
            pixels, left_metadata, depth,
            img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            source_type=imeta.ImageSourceType.REAL_WORLD,

            environment_type=imeta.EnvironmentType.OUTDOOR_URBAN,
            light_level=imeta.LightingLevel.POOR,
            time_of_day=imeta.TimeOfDay.NIGHT,

            red_mean=1000,
            blue_mean=1000,
            green_mean=1000,
            red_std=1000,
            green_std=1000,
            blue_std=1000,
            depth_mean=1000,
            depth_std=1000,

            simulation_world='TestSimulationWorld2',
            lighting_model=imeta.LightingModel.UNLIT,
            texture_mipmap_bias=2,
            normal_maps_enabled=False,
            roughness_enabled=False,
            geometry_decimation=0.2,
            minimum_object_volume=0.542,

            procedural_generation_seed=6743,
        )

        self.assertEqual(true_hash, bytes(right_metadata.img_hash))
        self.assertEqual(left_metadata.source_type, right_metadata.source_type)

        self.assertEqual(left_metadata.environment_type, right_metadata.environment_type)
        self.assertEqual(left_metadata.light_level, right_metadata.light_level)
        self.assertEqual(left_metadata.time_of_day, right_metadata.time_of_day)

        self.assertEqual(left_metadata.simulation_world, right_metadata.simulation_world)
        self.assertEqual(left_metadata.lighting_model, right_metadata.lighting_model)
        self.assertEqual(left_metadata.texture_mipmap_bias, right_metadata.texture_mipmap_bias)
        self.assertEqual(left_metadata.normal_maps_enabled, right_metadata.normal_maps_enabled)
        self.assertEqual(left_metadata.roughness_enabled, right_metadata.roughness_enabled)
        self.assertEqual(left_metadata.geometry_decimation, right_metadata.geometry_decimation)
        self.assertEqual(left_metadata.minimum_object_volume, right_metadata.minimum_object_volume)
        self.assertEqual(left_metadata.procedural_generation_seed, right_metadata.procedural_generation_seed)

        self.assertEqual(mean_red, right_metadata.red_mean)
        self.assertEqual(mean_green, right_metadata.green_mean)
        self.assertEqual(mean_blue, right_metadata.blue_mean)
        self.assertEqual(std_red, right_metadata.red_std)
        self.assertEqual(std_green, right_metadata.green_std)
        self.assertEqual(std_blue, right_metadata.blue_std)
        self.assertEqual(np.mean(depth), right_metadata.depth_mean)
        self.assertEqual(np.std(depth), right_metadata.depth_std)
