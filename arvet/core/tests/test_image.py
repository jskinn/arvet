# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
from pymodm.errors import ValidationError
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
        dbconn.setup_image_manager(mock=False)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        im.Image._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        im.Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_stores_and_loads_simple(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
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
            image_group='test',
            metadata=make_metadata(),
            additional_metadata={'test': True},
            depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            true_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
        )
        img.save()

        # Load all the entities
        all_entities = list(im.Image.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], img)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        # no pixels
        img = im.Image(
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

        # no metadata
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test'
        )
        with self.assertRaises(ValidationError):
            img.save()

        # no image group
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

    @unittest.skip("Not working at the moment, no overloaded Image delete")
    def test_deletes_pixel_data_when_deleted(self):
        group_name = 'test'
        image_manager = im_manager.get()
        pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.Image(
            pixels=pixels,
            image_group=group_name,
            metadata=imeta.make_metadata(pixels, source_type=imeta.ImageSourceType.SYNTHETIC),
            additional_metadata={'test': True}
        )
        img.save()

        with image_manager.get_group(group_name) as image_group:
            path = image_group.find_path_for_image(pixels)
            self.assertTrue(image_group.is_valid_path(path))
            im.Image.objects.all().delete()
            self.assertFalse(image_group.is_valid_path(path))

    @unittest.skip("Not working at the moment, no overloaded Image delete")
    def test_deletes_all_image_data_when_deleted(self):
        group_name = 'test'
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group=group_name,
            metadata=make_metadata(),
            additional_metadata={'test': True},
            depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            true_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
        )
        img.save()

        # Get the son image to find the image paths
        documents = list(im.Image.objects.all().values())
        self.assertEqual(len(documents), 1)
        paths = [
            documents[0]['pixel_path'],
            documents[0]['depth_path'],
            documents[0]['true_depth_path'],
            documents[0]['normals_path']
        ]

        image_manager = im_manager.get()
        with image_manager.get_group(group_name) as image_group:
            for path in paths:
                self.assertTrue(image_group.is_valid_path(path))

            # Delete all the images
            im.Image.objects.all().delete()
            for path in paths:
                self.assertFalse(image_group.is_valid_path(path))

    @unittest.skip("This is broken still, not using MaskedObjects currently")
    def test_deletes_object_masks_when_deleted(self):
        pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        group_name = 'test'
        img = im.Image(
            pixels=pixels,
            image_group=group_name,
            metadata=imeta.make_metadata(
                pixels=pixels,
                source_type=imeta.ImageSourceType.SYNTHETIC,
                labelled_objects=[
                    imeta.MaskedObject(
                        class_names=('class_1',),
                        x=152,
                        y=239,
                        mask=np.random.choice((True, False), size=(14, 78))
                    ),
                    imeta.MaskedObject(
                        class_names=('class_2',),
                        x=23,
                        y=12,
                        mask=np.random.choice((True, False), size=(14, 78))
                    )
                ]
            ),
        )
        img.save()

        # Get the son image to find the image paths
        documents = list(im.Image.objects.all().values())
        self.assertEqual(len(documents), 1)
        paths = [
            obj['mask']
            for obj in documents[0]['metadata']['labelled_objects']
        ]

        image_manager = im_manager.get()
        with image_manager.get_group(group_name) as image_group:
            for path in paths:
                self.assertTrue(image_group.is_valid_path(path))

            # Delete all the images
            im.Image.objects.all().delete()
            for path in paths:
                self.assertFalse(image_group.is_valid_path(path))


class TestImage(th.ExtendedTestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls):
        dbconn.tear_down_image_manager()

    def test_camera_pose(self):
        pose = tf.Transform((13, -22, 43), (1, 2, -4, 2))
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
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

    def test_hash(self):
        img_hash = b'\x1f`\xa8\x8aR\xed\x9f\x0b'
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=img_hash,
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertEqual(img_hash, bytes(img.hash))

    def test_pixels_retrieves_data(self):
        pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.Image(
            pixels=pixels,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(pixels, img.pixels)

    def test_depth_retrieves_data(self):
        depth = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            depth=depth,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(depth, img.depth)

    def test_depth_is_none_when_omitted(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.depth)

    def test_true_depth_retrieves_data(self):
        true_depth = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            true_depth=true_depth,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(true_depth, img.true_depth)

    def test_true_depth_is_none_when_omitted(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.true_depth)

    def test_normals_retrieves_data(self):
        normals = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            normals=normals,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(normals, img.normals)

    def test_normals_is_none_when_omitted(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.normals)

    def test_get_columns_returns_column_list(self):
        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertEqual({
            'pixel_path',
            'image_group',
            'source_type',
            'lens_focal_distance',
            'aperture',

            'pos_x',
            'pos_y',
            'pos_z',

            'red_mean',
            'red_std',
            'green_mean',
            'green_std',
            'blue_mean',
            'blue_std',
            'depth_mean',
            'depth_std',

            'environment_type',
            'light_level',
            'time_of_day',

            'simulation_world',
            'lighting_model',
            'texture_mipmap_bias',
            'normal_maps_enabled',
            'roughness_enabled',
            'geometry_decimation',
        }, img.get_columns())

    def test_get_properties_returns_the_value_of_all_columns(self):
        image_group = 'my-test-group'
        source_type = imeta.ImageSourceType.SYNTHETIC
        lens_focal_distance = np.random.uniform(10, 1000)
        aperture = np.random.uniform(1.2, 3.6)

        pos_x = np.random.uniform(-1000, 1000)
        pos_y = np.random.uniform(-1000, 1000)
        pos_z = np.random.uniform(-1000, 1000)

        red_mean = np.random.uniform(10, 240)
        red_std = np.random.uniform(1, 10)
        green_mean = np.random.uniform(10, 240)
        green_std = np.random.uniform(1, 10)
        blue_mean = np.random.uniform(10, 240)
        blue_std = np.random.uniform(1, 10)
        depth_mean = np.random.uniform(100, 2000)
        depth_std = np.random.uniform(10, 100)

        environment_type = imeta.EnvironmentType.OUTDOOR_URBAN
        light_level = imeta.LightingLevel.DIM
        time_of_day = imeta.TimeOfDay.TWILIGHT

        simulation_world = "Generated World {0}".format(np.random.randint(0, 2**16))
        lighting_model = imeta.LightingModel.UNLIT
        texture_mipmap_bias = np.random.randint(1, 8)
        normal_maps_enabled = False
        roughness_enabled = True
        geometry_decimation = np.random.uniform(0, 1)

        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group=image_group,
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                camera_pose=tf.Transform(location=[pos_x, pos_y, pos_z]),
                source_type=source_type,
                lens_focal_distance=lens_focal_distance,
                aperture=aperture,
                red_mean=red_mean,
                red_std=red_std,
                green_mean=green_mean,
                green_std=green_std,
                blue_mean=blue_mean,
                blue_std=blue_std,
                depth_mean=depth_mean,
                depth_std=depth_std,
                environment_type=environment_type,
                light_level=light_level,
                time_of_day=time_of_day,
                simulation_world=simulation_world,
                lighting_model=lighting_model,
                texture_mipmap_bias=texture_mipmap_bias,
                normal_maps_enabled=normal_maps_enabled,
                roughness_enabled=roughness_enabled,
                geometry_decimation=geometry_decimation
            )
        )
        self.assertEqual({
            'pixel_path': img.pixel_path,
            'image_group': image_group,
            'source_type': source_type,
            'lens_focal_distance': lens_focal_distance,
            'aperture': aperture,

            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,

            'red_mean': red_mean,
            'red_std': red_std,
            'green_mean': green_mean,
            'green_std': green_std,
            'blue_mean': blue_mean,
            'blue_std': blue_std,
            'depth_mean': depth_mean,
            'depth_std': depth_std,

            'environment_type': environment_type,
            'light_level': light_level.value,
            'time_of_day': time_of_day,

            'simulation_world': simulation_world,
            'lighting_model': lighting_model,
            'texture_mipmap_bias': texture_mipmap_bias,
            'normal_maps_enabled': normal_maps_enabled,
            'roughness_enabled': roughness_enabled,
            'geometry_decimation': geometry_decimation
        }, img.get_properties())

    def test_get_properties_returns_defaults_for_minimal_parameters(self):
        source_type = imeta.ImageSourceType.SYNTHETIC

        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=source_type
            )
        )
        self.assertEqual({
            'pixel_path': img.pixel_path,
            'image_group': 'test',
            'source_type': source_type,
            'lens_focal_distance': None,
            'aperture': None,

            'pos_x': None,
            'pos_y': None,
            'pos_z': None,

            'red_mean': None,
            'red_std': None,
            'green_mean': None,
            'green_std': None,
            'blue_mean': None,
            'blue_std': None,
            'depth_mean': None,
            'depth_std': None,

            'environment_type': None,
            'light_level': None,
            'time_of_day': None,

            'simulation_world': None,
            'lighting_model': None,
            'texture_mipmap_bias': None,
            'normal_maps_enabled': None,
            'roughness_enabled': None,
            'geometry_decimation': None
        }, img.get_properties())

    def test_get_properties_returns_only_requested_columns_that_exist(self):
        source_type = imeta.ImageSourceType.SYNTHETIC
        lens_focal_distance = np.random.uniform(10, 1000)
        aperture = np.random.uniform(1.2, 3.6)

        red_mean = np.random.uniform(10, 240)
        red_std = np.random.uniform(1, 10)
        green_mean = np.random.uniform(10, 240)
        green_std = np.random.uniform(1, 10)
        blue_mean = np.random.uniform(10, 240)
        blue_std = np.random.uniform(1, 10)
        depth_mean = np.random.uniform(100, 2000)
        depth_std = np.random.uniform(10, 100)

        environment_type = imeta.EnvironmentType.OUTDOOR_URBAN
        light_level = imeta.LightingLevel.DIM
        time_of_day = imeta.TimeOfDay.TWILIGHT

        simulation_world = "Generated World {0}".format(np.random.randint(0, 2**16))
        lighting_model = imeta.LightingModel.UNLIT
        texture_mipmap_bias = np.random.randint(1, 8)
        normal_maps_enabled = False
        roughness_enabled = True
        geometry_decimation = np.random.uniform(0, 1)

        img = im.Image(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=source_type,
                lens_focal_distance=lens_focal_distance,
                aperture=aperture,
                red_mean=red_mean,
                red_std=red_std,
                green_mean=green_mean,
                green_std=green_std,
                blue_mean=blue_mean,
                blue_std=blue_std,
                depth_mean=depth_mean,
                depth_std=depth_std,
                environment_type=environment_type,
                light_level=light_level,
                time_of_day=time_of_day,
                simulation_world=simulation_world,
                lighting_model=lighting_model,
                texture_mipmap_bias=texture_mipmap_bias,
                normal_maps_enabled=normal_maps_enabled,
                roughness_enabled=roughness_enabled,
                geometry_decimation=geometry_decimation
            )
        )
        self.assertEqual({
            'red_mean': red_mean,
            'red_std': red_std,
            'green_mean': green_mean,
            'green_std': green_std,
            'blue_mean': blue_mean,
            'blue_std': blue_std,
            'depth_mean': depth_mean,
            'depth_std': depth_std
        }, img.get_properties({
            'red_mean',
            'red_std',
            'green_mean',
            'green_std',
            'blue_mean',
            'blue_std',
            'depth_mean',
            'depth_std',
            'not_a_real_column',
            'another_not_a_real_column'
        }))


class TestStereoImageDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager(mock=False)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        im.Image._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        im.Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_stores_and_loads_simple(self):
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            ),
            right_metadata=imeta.ImageMetadata(
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
            image_group='test',
            metadata=make_metadata(),
            right_metadata=make_metadata(img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0'),
            additional_metadata={'test': True},
            depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            right_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            true_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            right_true_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
        )
        img.save()

        # Load all the entities
        all_entities = list(im.Image.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], img)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        # no pixels
        img = im.StereoImage(
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            ),
            right_metadata=imeta.ImageMetadata(
                img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

        # no metadata
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            right_metadata=imeta.ImageMetadata(
                img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

        # no right pixels
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            ),
            right_metadata=imeta.ImageMetadata(
                img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

        # no right metadata
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

        # no image group
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            ),
            right_metadata=imeta.ImageMetadata(
                img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        with self.assertRaises(ValidationError):
            img.save()

    @unittest.skip("Not working, no overloaded image delete")
    def test_deletes_pixel_data_when_deleted(self):
        group_name = 'test'
        image_manager = im_manager.get()
        left_pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        right_pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        metadata = imeta.make_metadata(left_pixels, source_type=imeta.ImageSourceType.SYNTHETIC)
        img = im.StereoImage(
            pixels=left_pixels,
            right_pixels=right_pixels,
            image_group=group_name,
            metadata=metadata,
            right_metadata=imeta.make_right_metadata(right_pixels, metadata),
            additional_metadata={'test': True}
        )
        img.save()
        with image_manager.get_group(group_name) as image_group:
            left_path = image_group.find_path_for_image(left_pixels)
            right_path = image_group.find_path_for_image(left_pixels)

            self.assertTrue(image_group.is_valid_path(left_path))
            self.assertTrue(image_group.is_valid_path(right_path))
            im.StereoImage.objects.all().delete()
            self.assertFalse(image_group.is_valid_path(left_path))
            self.assertFalse(image_group.is_valid_path(right_path))

    @unittest.skip("Not working, no overloaded image delete")
    def test_deletes_all_image_data_when_deleted(self):
        group_name = 'test'
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group=group_name,
            metadata=make_metadata(),
            right_metadata=make_metadata(img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0'),
            additional_metadata={'test': True},
            depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            right_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            true_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            right_true_depth=np.random.uniform(0.1, 7.1, size=(100, 100)),
            normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_normals=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
        )
        img.save()

        # Get the son image to find the image paths
        documents = list(im.StereoImage.objects.all().values())
        self.assertEqual(len(documents), 1)
        paths = [
            documents[0]['pixels'],
            documents[0]['right_pixels'],
            documents[0]['depth'],
            documents[0]['right_depth'],
            documents[0]['true_depth'],
            documents[0]['right_true_depth'],
            documents[0]['normals'],
            documents[0]['right_normals']
        ]

        image_manager = im_manager.get()
        with image_manager.get_group(group_name) as image_group:
            for path in paths:
                self.assertTrue(image_group.is_valid_path(path))

            # Delete all the images
            im.StereoImage.objects.all().delete()
            for path in paths:
                self.assertFalse(image_group.is_valid_path(path))

    @unittest.skip("This is broken still, not using MaskedObjects currently")
    def test_deletes_object_masks_when_deleted(self):
        pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        right_pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        group_name = 'test'
        left_metadata = imeta.make_metadata(
            pixels=pixels,
            source_type=imeta.ImageSourceType.SYNTHETIC,
            labelled_objects=[
                imeta.MaskedObject(
                    class_names=('class_1',),
                    x=152,
                    y=239,
                    mask=np.random.choice((True, False), size=(14, 78))
                ),
                imeta.MaskedObject(
                    class_names=('class_2',),
                    x=23,
                    y=12,
                    mask=np.random.choice((True, False), size=(14, 78))
                )
            ]
        )
        right_metadata = imeta.make_right_metadata(
            pixels=right_pixels,
            left_metadata=left_metadata,
            labelled_objects=[
                imeta.MaskedObject(
                    class_names=('class_1',),
                    x=151,
                    y=29,
                    mask=np.random.choice((True, False), size=(14, 78))
                ),
                imeta.MaskedObject(
                    class_names=('class_2',),
                    x=213,
                    y=26,
                    mask=np.random.choice((True, False), size=(14, 78))
                )
            ]
        )
        img = im.StereoImage(
            pixels=pixels,
            right_pixels=right_pixels,
            image_group=group_name,
            metadata=left_metadata,
            right_metadata=right_metadata
        )
        img.save()

        # Get the son image to find the image paths
        documents = list(im.StereoImage.objects.all().values())
        self.assertEqual(len(documents), 1)
        paths = [
            obj['mask']
            for obj in documents[0]['metadata']['labelled_objects']
        ]
        paths.extend(
            obj['mask']
            for obj in documents[0]['right_metadata']['labelled_objects']
        )

        image_manager = im_manager.get()
        with image_manager.get_group(group_name) as image_group:
            for path in paths:
                self.assertTrue(image_group.is_valid_path(path))

            # Delete all the images
            im.StereoImage.objects.all().delete()
            for path in paths:
                self.assertFalse(image_group.is_valid_path(path))


class TestStereoImage(th.ExtendedTestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls):
        dbconn.tear_down_image_manager()

    def setUp(self):
        self.image_group = 'test'
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
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)),
                                     instance_name='Car-002'),
                imeta.LabelledObject(class_names=('cat',), x=125, y=244, width=117, height=67,
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     instance_name='cat-090')
            ))

        self.left_pixels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.right_pixels = np.asarray(np.random.uniform(0, 255, (32, 32, 3)), dtype='uint8')
        self.image = im.StereoImage(pixels=self.left_pixels,
                                    right_pixels=self.right_pixels,
                                    image_group=self.image_group,
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
            image_group=self.image_group,
            depth=self.left_depth,
            right_depth=self.right_depth,
            true_depth=self.left_gt_depth,
            right_true_depth=self.right_gt_depth,
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

    def test_true_depth_data(self):
        self.assertEqual(self.image.left_true_depth, None)
        self.assertEqual(self.image.right_true_depth, None)
        self.assertNPEqual(self.full_image.left_true_depth, self.left_gt_depth)
        self.assertNPEqual(self.full_image.right_true_depth, self.right_gt_depth)

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

    def test_hash(self):
        # Hash is hard-coded in SetUp
        self.assertEqual(b'\x1f`\xa8\x8aR\xed\x9f\x0b', bytes(self.image.hash))
        self.assertEqual(b'\xa5\xc9\x08\xaf$\x0b\x116', bytes(self.image.right_hash))

    def test_right_pixels_retrieves_data(self):
        right_pixels = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.StereoImage(
            right_pixels=right_pixels,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(right_pixels, img.right_pixels)

    def test_right_depth_retrieves_data(self):
        right_depth = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.StereoImage(
            right_depth=right_depth,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(right_depth, img.right_depth)

    def test_right_depth_is_none_when_omitted(self):
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.right_depth)

    def test_right_true_depth_retrieves_data(self):
        right_true_depth = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_true_depth=right_true_depth,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(right_true_depth, img.right_true_depth)

    def test_right_true_depth_is_none_when_omitted(self):
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.right_true_depth)

    def test_normals_retrieves_data(self):
        right_normals = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            right_normals=right_normals,
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertNPEqual(right_normals, img.right_normals)

    def test_normals_is_none_when_omitted(self):
        img = im.StereoImage(
            pixels=np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8),
            image_group='test',
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertIsNone(img.right_normals)

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
            image_group=self.image_group,
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
            image_group='right_image_group',
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
        self.assertEqual(stereo_image.image_group, left_image.image_group)
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

    def test_get_columns_returns_column_list(self):
        self.assertEqual({
            # Left columns
            'pixel_path',
            'image_group',
            'source_type',
            'lens_focal_distance',
            'aperture',

            'pos_x',
            'pos_y',
            'pos_z',

            'red_mean',
            'red_std',
            'green_mean',
            'green_std',
            'blue_mean',
            'blue_std',
            'depth_mean',
            'depth_std',

            'environment_type',
            'light_level',
            'time_of_day',

            'simulation_world',
            'lighting_model',
            'texture_mipmap_bias',
            'normal_maps_enabled',
            'roughness_enabled',
            'geometry_decimation',

            # Right columns
            'stereo_offset',
            'right_lens_focal_distance',
            'right_aperture',

            'right_red_mean',
            'right_red_std',
            'right_green_mean',
            'right_green_std',
            'right_blue_mean',
            'right_blue_std',
            'right_depth_mean',
            'right_depth_std'
        }, self.image.get_columns())

    def test_get_properties_returns_the_value_of_all_columns(self):
        self.assertEqual({
            # Left image properties
            'pixel_path': self.full_image.pixel_path,
            'image_group': self.image_group,
            'source_type': imeta.ImageSourceType.SYNTHETIC,
            'lens_focal_distance': 5.0,
            'aperture': 22,

            'pos_x': self.full_left_pose.location[0],
            'pos_y': self.full_left_pose.location[1],
            'pos_z': self.full_left_pose.location[2],

            'red_mean': self.full_metadata.red_mean,
            'red_std': self.full_metadata.red_std,
            'green_mean': self.full_metadata.green_mean,
            'green_std': self.full_metadata.green_std,
            'blue_mean': self.full_metadata.blue_mean,
            'blue_std': self.full_metadata.blue_std,
            'depth_mean': self.full_metadata.depth_mean,
            'depth_std': self.full_metadata.depth_std,

            'environment_type': imeta.EnvironmentType.INDOOR_CLOSE,
            'light_level': imeta.LightingLevel.WELL_LIT.value,
            'time_of_day': imeta.TimeOfDay.DAY,

            'simulation_world': 'TestSimulationWorld',
            'lighting_model': imeta.LightingModel.LIT,
            'texture_mipmap_bias': 1,
            'normal_maps_enabled': True,
            'roughness_enabled': True,
            'geometry_decimation': 0.8,

            # Right columns
            'stereo_offset': np.linalg.norm(self.full_image.stereo_offset.location),
            'right_lens_focal_distance': self.full_image.right_metadata.lens_focal_distance,
            'right_aperture': self.full_image.right_metadata.aperture,

            'right_red_mean': self.full_image.right_metadata.red_mean,
            'right_red_std': self.full_image.right_metadata.red_std,
            'right_green_mean': self.full_image.right_metadata.green_mean,
            'right_green_std': self.full_image.right_metadata.green_std,
            'right_blue_mean': self.full_image.right_metadata.blue_mean,
            'right_blue_std': self.full_image.right_metadata.blue_std,
            'right_depth_mean': self.full_image.right_metadata.depth_mean,
            'right_depth_std': self.full_image.right_metadata.depth_std

        }, self.full_image.get_properties())

    def test_get_properties_returns_defaults_for_minimal_parameters(self):
        image_group = 'test-images-6'
        source_type = imeta.ImageSourceType.SYNTHETIC

        img = im.StereoImage(
            pixels=self.left_pixels,
            right_pixels=self.right_pixels,
            image_group=image_group,
            metadata=imeta.ImageMetadata(
                img_hash=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
                source_type=imeta.ImageSourceType.SYNTHETIC
            ),
            right_metadata=imeta.ImageMetadata(
                img_hash=b'\x3a`\x8a\xa8H\xde\xf9\xb0',
                source_type=imeta.ImageSourceType.SYNTHETIC
            )
        )
        self.assertEqual({
            # Left image properties
            'pixel_path': img.pixel_path,
            'image_group': image_group,
            'source_type': source_type,
            'lens_focal_distance': None,
            'aperture': None,

            'pos_x': None,
            'pos_y': None,
            'pos_z': None,

            'red_mean': None,
            'red_std': None,
            'green_mean': None,
            'green_std': None,
            'blue_mean': None,
            'blue_std': None,
            'depth_mean': None,
            'depth_std': None,

            'environment_type': None,
            'light_level': None,
            'time_of_day': None,

            'simulation_world': None,
            'lighting_model': None,
            'texture_mipmap_bias': None,
            'normal_maps_enabled': None,
            'roughness_enabled': None,
            'geometry_decimation': None,

            # Right columns
            'stereo_offset': None,
            'right_lens_focal_distance': None,
            'right_aperture': None,

            'right_red_mean': None,
            'right_red_std': None,
            'right_green_mean': None,
            'right_green_std': None,
            'right_blue_mean': None,
            'right_blue_std': None,
            'right_depth_mean': None,
            'right_depth_std': None
        }, img.get_properties())

    def test_get_properties_returns_only_requested_columns_that_exist(self):
        self.assertEqual({
            # Left image properties
            'source_type': imeta.ImageSourceType.SYNTHETIC,
            'lens_focal_distance': 5.0,
            'aperture': 22.0,

            'environment_type': imeta.EnvironmentType.INDOOR_CLOSE,
            'light_level': imeta.LightingLevel.WELL_LIT.value,
            'time_of_day': imeta.TimeOfDay.DAY,

            'simulation_world': 'TestSimulationWorld',
            'lighting_model': imeta.LightingModel.LIT,
            'texture_mipmap_bias': 1,
            'normal_maps_enabled': True,
            'roughness_enabled': True,
            'geometry_decimation': 0.8,

            # Right columns
            'stereo_offset': np.linalg.norm(self.full_image.stereo_offset.location),
            'right_lens_focal_distance': self.full_image.right_metadata.lens_focal_distance,
            'right_aperture': self.full_image.right_metadata.aperture

        }, self.full_image.get_properties({
            'source_type',
            'lens_focal_distance',
            'aperture',

            'environment_type',
            'light_level',
            'time_of_day',

            'not_a_real_column',
            'another_not_real_column',

            'simulation_world',
            'lighting_model',
            'texture_mipmap_bias',
            'normal_maps_enabled',
            'roughness_enabled',
            'geometry_decimation',

            'stereo_offset',
            'right_lens_focal_distance',
            'right_aperture',
        }))
