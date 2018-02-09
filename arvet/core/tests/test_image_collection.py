# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import arvet.database.tests.test_entity
import numpy as np
import bson.objectid
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import arvet.core.image_entity as ie
import arvet.core.image_collection as ic
import arvet.core.sequence_type


def make_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'id_': bson.objectid.ObjectId(),
        'data': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': imeta.ImageMetadata(
            hash_=b'\xf1\x9a\xe2|' + np.random.randint(0, 0xFFFFFFFF).to_bytes(4, 'big'),
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=tf.Transform(location=(1 + 100*index, 2 + np.random.uniform(-1, 1), 3),
                                     rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
            intrinsics=cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=(
                imeta.LabelledObject(class_names=('car',), bounding_box=(12, 144, 67, 43), label_color=(123, 127, 112),
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)), object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',), bounding_box=(125, 244, 117, 67), label_color=(27, 89, 62),
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     object_id='cat-090')
            ), average_scene_depth=90.12),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth_data': np.random.uniform(0, 1, (32, 32)),
        'labels_data': np.random.uniform(0, 1, (32, 32, 3)),
        'world_normals_data': np.random.uniform(0, 1, (32, 32, 3))
    })
    return ie.ImageEntity(**kwargs)


def make_stereo_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'id_': bson.objectid.ObjectId(),
        'left_data': np.random.uniform(0, 255, (32, 32, 3)),
        'right_data': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': imeta.ImageMetadata(
            hash_=b'\xf1\x9a\xe2|' + np.random.randint(0, 0xFFFFFFFF).to_bytes(4, 'big'),
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=tf.Transform(location=(1 + 100*index, 2 + np.random.uniform(-1, 1), 3),
                                     rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
            right_camera_pose=tf.Transform(location=(1 + 100*index, 12 + np.random.uniform(-1, 1), 3),
                                           rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
            intrinsics=cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
            right_intrinsics=cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=(
                imeta.LabelledObject(class_names=('car',), bounding_box=(12, 144, 67, 43), label_color=(123, 127, 112),
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)), object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',), bounding_box=(125, 244, 117, 67), label_color=(27, 89, 62),
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     object_id='cat-090')
            ), average_scene_depth=90.12),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'left_depth_data': np.random.uniform(0, 1, (32, 32)),
        'right_depth_data': np.random.uniform(0, 1, (32, 32)),
        'left_labels_data': np.random.uniform(0, 1, (32, 32, 3)),
        'right_labels_data': np.random.uniform(0, 1, (32, 32, 3)),
        'left_world_normals_data': np.random.uniform(0, 1, (32, 32, 3)),
        'right_world_normals_data': np.random.uniform(0, 1, (32, 32, 3))
    })
    return ie.StereoImageEntity(**kwargs)


class TestImageCollection(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.image_map = {}
        self.images = {}
        for i in range(10):
            image = make_image(i)
            self.image_map[image.identifier] = image
            self.images[i * 1.2] = image.identifier

    def get_class(self):
        return ic.ImageCollection

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'images': self.images,
            'type_': arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
            'db_client_': self.create_mock_db_client()
        })
        return ic.ImageCollection(*args, **kwargs)

    def assert_models_equal(self, image_collection1, image_collection2):
        """
        Helper to assert that two image entities are equal
        :param image_collection1: ImageEntity
        :param image_collection2: ImageEntity
        :return:
        """
        if (not isinstance(image_collection1, ic.ImageCollection) or
                not isinstance(image_collection2, ic.ImageCollection)):
            self.fail('object was not an Image Collection')
        self.assertEqual(image_collection1.identifier, image_collection2.identifier)
        self.assertEqual(image_collection1.sequence_type, image_collection2.sequence_type)
        self.assertEqual(image_collection1.is_depth_available, image_collection2.is_depth_available)
        self.assertEqual(image_collection1.is_per_pixel_labels_available,
                         image_collection2.is_per_pixel_labels_available)
        self.assertEqual(image_collection1.is_normals_available, image_collection2.is_normals_available)
        self.assertEqual(image_collection1.is_stereo_available, image_collection2.is_stereo_available)
        self.assertEqual(len(image_collection1), len(image_collection2))
        self.assertEqual(image_collection1.timestamps, image_collection2.timestamps)
        with image_collection1:
            with image_collection2:
                for idx in range(len(image_collection1)):
                    img1, stamp1 = image_collection1.get_next_image()
                    img2, stamp2 = image_collection2.get_next_image()
                    self.assertEqual(stamp1, stamp2)
                    self.assertEqual(img1.identifier, img2.identifier)
                    self.assertTrue(np.array_equal(img1.data, img2.data))
                    self.assertEqual(img1.camera_pose, img2.camera_pose)
                    self.assertTrue(np.array_equal(img1.depth_data, img2.depth_data))
                    self.assertTrue(np.array_equal(img1.labels_data, img2.labels_data))
                    self.assertTrue(np.array_equal(img1.world_normals_data, img2.world_normals_data))
                    self.assertEqual(img1.metadata, img2.metadata)
                    self.assertEqual(img1.additional_metadata, img2.additional_metadata)
        self.assertTrue(image_collection1.is_complete())
        self.assertTrue(image_collection2.is_complete())

    def create_mock_db_client(self):
        db_client = super().create_mock_db_client()

        # Add our images to the mock image collection
        for image in self.image_map.values():
            image.save_image_data(db_client)
            db_client.image_collection.insert(image.serialize())

        db_client.deserialize_entity.side_effect = lambda s_image: self.image_map[s_image['_id']]
        return db_client

    def test_timestamps_returns_all_timestamps_in_order(self):
        subject = ic.ImageCollection(images=self.images, type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     db_client_=self.create_mock_db_client())
        self.assertEqual([1.2 * t for t in range(10)], subject.timestamps)
        for stamp in subject.timestamps:
            self.assertIsNotNone(subject.get(stamp))

    def test_is_depth_available_is_true_iff_all_images_have_depth_data(self):
        db_client = self.create_mock_db_client()
        subject = ic.ImageCollection(images=self.images, type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     db_client_=db_client)
        self.assertTrue(subject.is_depth_available)

        image = make_image(depth_data=None)
        image.save_image_data(db_client)
        db_client.image_collection.insert(image.serialize())
        subject = ic.ImageCollection(type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     images=du.defaults({1.7: image.identifier}, self.images), db_client_=db_client)
        self.assertFalse(subject.is_depth_available)

    def test_is_per_pixel_labels_available_is_true_iff_all_images_have_labels_data(self):
        db_client = self.create_mock_db_client()
        subject = ic.ImageCollection(images=self.images, type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     db_client_=db_client)
        self.assertTrue(subject.is_per_pixel_labels_available)

        image = make_image(labels_data=None)
        image.save_image_data(db_client)
        db_client.image_collection.insert(image.serialize())
        subject = ic.ImageCollection(type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     images=du.defaults({1.7: image.identifier}, self.images), db_client_=db_client)
        self.assertFalse(subject.is_per_pixel_labels_available)

    def test_is_labels_available_is_true_iff_all_images_have_bounding_boxes(self):
        db_client = self.create_mock_db_client()
        subject = ic.ImageCollection(images=self.images, type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     db_client_=db_client)
        self.assertTrue(subject.is_labels_available)

        image = make_image(metadata=imeta.ImageMetadata(
            hash_=b'\xf1\x9a\xe2|' + np.random.randint(0, 0xFFFFFFFF).to_bytes(4, 'big'),
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=tf.Transform(location=(800, 2 + np.random.uniform(-1, 1), 3),
                                     rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
            intrinsics=cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=[],
            average_scene_depth=90.12
        ))
        image.save_image_data(db_client)
        db_client.image_collection.insert(image.serialize())
        subject = ic.ImageCollection(type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     images=du.defaults({1.7: image.identifier}, self.images), db_client_=db_client)
        self.assertFalse(subject.is_labels_available)

    def test_is_normals_available_is_true_iff_all_images_have_normals_data(self):
        db_client = self.create_mock_db_client()
        subject = ic.ImageCollection(images=self.images, type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     db_client_=db_client)
        self.assertTrue(subject.is_normals_available)

        image = make_image(world_normals_data=None)
        image.save_image_data(db_client)
        db_client.image_collection.insert(image.serialize())
        subject = ic.ImageCollection(type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     images=du.defaults({1.7: image.identifier}, self.images), db_client_=db_client)
        self.assertFalse(subject.is_normals_available)

    def test_is_stereo_available_is_true_iff_all_images_are_stereo_images(self):
        # Create some stereo images and add them to the map
        stereo_images_list = {}
        for i in range(10):
            stereo_image = make_stereo_image(index=i)
            self.image_map[stereo_image.identifier] = stereo_image
            stereo_images_list[i * 1.3] = stereo_image.identifier
        db_client = self.create_mock_db_client()
        subject = ic.ImageCollection(type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     images=stereo_images_list,
                                     db_client_=db_client)
        self.assertTrue(subject.is_stereo_available)
        subject = ic.ImageCollection(type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                                     images=du.defaults(stereo_images_list, self.images),
                                     db_client_=self.create_mock_db_client())
        self.assertFalse(subject.is_stereo_available)

    def test_get_next_image_returns_images_in_order(self):
        subject = self.make_instance()
        timestamps = sorted(self.images.keys())
        for stamp in timestamps:
            result_image, result_timestamp = subject.get_next_image()
            self.assertEqual(stamp, result_timestamp)
            self.assertEqual(self.images[stamp], result_image.identifier)
            self.assertTrue(np.array_equal(self.image_map[self.images[stamp]].data, result_image.data))
        self.assertTrue(subject.is_complete())
        self.assertEqual((None, None), subject.get_next_image())

    def test_begin_restarts(self):
        subject = self.make_instance()
        subject.begin()
        subject.get_next_image()
        subject.get_next_image()
        subject.get_next_image()

        subject.begin()
        for stamp in sorted(self.images.keys()):
            result_image, timestamp = subject.get_next_image()
            self.assertEqual(stamp, timestamp)
            self.assertEqual(self.images[stamp], result_image.identifier)
            self.assertTrue(np.array_equal(self.image_map[self.images[stamp]].data, result_image.data))
        self.assertTrue(subject.is_complete())
        self.assertEqual((None, None), subject.get_next_image())

    def test_loads_images_on_the_fly(self):
        db_client = self.create_mock_db_client()
        subject = self.make_instance(db_client_=db_client)

        with subject:
            while not subject.is_complete():
                image, stamp = subject.get_next_image()
                # Check that we loaded the image
                self.assertIn(mock.call({'_id': image.identifier}), db_client.image_collection.find_one.call_args_list)

    def test_create_and_save_checks_image_ids_and_stops_if_not_all_found(self):
        db_client = self.create_mock_db_client()
        mock_cursor = mock.MagicMock()
        mock_cursor.count.return_value = len(self.image_map) - 2    # Return missing ids.
        db_client.image_collection.find.return_value = mock_cursor

        result = ic.ImageCollection.create_and_save(
            db_client, {timestamp: image_id for timestamp, image_id in self.images.items()},
            arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertIsNone(result)
        self.assertTrue(db_client.image_collection.find.called)
        self.assertEqual({'_id': {'$in': [image_id for image_id in self.images.values()]}},
                         db_client.image_collection.find.call_args[0][0])
        self.assertFalse(db_client.image_source_collection.insert.called)

    def test_create_and_save_checks_for_existing_collection(self):
        db_client = self.create_mock_db_client()
        ic.ImageCollection.create_and_save(db_client,
                                           {timestamp: image_id for timestamp, image_id in self.images.items()},
                                           arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)

        self.assertTrue(db_client.image_source_collection.find_one.called)
        existing_query = db_client.image_source_collection.find_one.call_args[0][0]
        self.assertIn('_type', existing_query)
        self.assertEqual('arvet.core.image_collection.ImageCollection', existing_query['_type'])
        self.assertIn('sequence_type', existing_query)
        self.assertEqual('SEQ', existing_query['sequence_type'])
        self.assertIn('images', existing_query)
        self.assertIn('$all', existing_query['images'])
        # Because dicts, we can't guarantee the order of this list
        # So we use $all, and make sure all the timestamp->image_id pairs are in it
        for timestamp, image_id in self.images.items():
            self.assertIn((timestamp, image_id), existing_query['images']['$all'])

    def test_create_and_save_makes_valid_collection(self):
        db_client = self.create_mock_db_client()
        mock_cursor = mock.MagicMock()
        mock_cursor.count.return_value = len(self.image_map)
        db_client.image_collection.find.return_value = mock_cursor
        db_client.image_source_collection.find_one.return_value = None

        ic.ImageCollection.create_and_save(db_client,
                                           {timestamp: image_id for timestamp, image_id in self.images.items()},
                                           arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(db_client.image_source_collection.insert.called)
        s_image_collection = db_client.image_source_collection.insert.call_args[0][0]

        db_client = self.create_mock_db_client()
        collection = ic.ImageCollection.deserialize(s_image_collection, db_client)
        self.assertEqual(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL, collection.sequence_type)
        self.assertEqual(len(self.images), len(collection))
        for stamp, image_id in self.images.items():
            image = self.image_map[image_id]
            self.assertEqual(image.identifier, collection[stamp].identifier)
            self.assertTrue(np.array_equal(image.data, collection[stamp].data))
            self.assertEqual(image.camera_pose, collection[stamp].camera_pose)
            self.assertTrue(np.array_equal(image.depth_data, collection[stamp].depth_data))
            self.assertTrue(np.array_equal(image.labels_data, collection[stamp].labels_data))
            self.assertTrue(np.array_equal(image.world_normals_data, collection[stamp].world_normals_data))
            self.assertEqual(image.additional_metadata, collection[stamp].additional_metadata)

        s_image_collection_2 = collection.serialize()
        self.assert_serialized_equal(s_image_collection, s_image_collection_2)