# Copyright (c) 2017, John Skinner
import unittest
import os
from operator import itemgetter
import numpy as np
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import arvet.core.image as im
import arvet.core.image_collection as ic
from arvet.core.sequence_type import ImageSequenceType


class TestImageCollectionDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        ic.ImageCollection._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        im.Image._mongometa.collection.drop()
        ic.ImageCollection._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads_mono_kwargs(self):
        images = []
        for idx in range(10):
            img = make_image(idx)
            img.save()
            images.append(img)

        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.1 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL,
            is_depth_available=True,
            is_normals_available=True,
            is_stereo_available=False,
            is_labels_available=True,
            is_masks_available=False,
            camera_intrinsics=images[0].metadata.intrinsics
        )
        collection.save()

        # Load all the entities
        all_entities = list(ic.ImageCollection.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], collection)
        all_entities[0].delete()

    def test_stores_and_loads_mono_args(self):
        images = []
        for idx in range(10):
            img = make_image(idx)
            img.save()
            images.append(img)

        collection = ic.ImageCollection(
            images,
            [1.1 * idx for idx in range(10)],
            ImageSequenceType.SEQUENTIAL,
            True,
            True,
            False,
            True,
            False,
            images[0].metadata.intrinsics
        )
        collection.save()

        # Load all the entities
        all_entities = list(ic.ImageCollection.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], collection)
        all_entities[0].delete()

    def test_stores_and_loads_stereo(self):
        images = []
        for idx in range(10):
            img = make_stereo_image(idx)
            img.save()
            images.append(img)

        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.1 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL,
            is_depth_available=True,
            is_normals_available=True,
            is_stereo_available=False,
            is_labels_available=True,
            is_masks_available=False,
            camera_intrinsics=images[0].metadata.intrinsics,
            right_camera_pose=images[0].right_camera_pose,
            right_camera_intrinsics=images[0].right_metadata.intrinsics
        )
        collection.save()

        # Load all the entities
        all_entities = list(ic.ImageCollection.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], collection)
        all_entities[0].delete()

    def test_can_be_compared_without_loading_images(self):
        images = []
        for idx in range(10):
            img = CountedImage(
                pixels=np.random.uniform(0, 255, (32, 32, 3)),
                metadata=make_metadata(idx),
            )
            img.save()
            images.append(img)

        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.1 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        collection.save()
        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.2 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        collection.save()
        CountedImage.instance_count = 0

        # Load all the entities
        all_entities = list(ic.ImageCollection.objects.all())
        self.assertEqual(len(all_entities), 2)
        self.assertNotEqual(all_entities[0], all_entities[1])
        self.assertEqual(CountedImage.instance_count, 0)

        # Check actually reading the images loads them
        _ = all_entities[0].images[0]
        self.assertEqual(CountedImage.instance_count, 10)

        all_entities[0].delete()
        all_entities[1].delete()


class CountedImage(im.Image):
    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountedImage.instance_count += 1


class TestImageCollection(unittest.TestCase):

    def test_constructor_works_with_mixed_args(self):
        images = [make_image(idx, depth=None) for idx in range(10)]

        arguments = [
            ('images', images),
            ('timestamps', [1.1 * idx for idx in range(10)]),
            ('sequence_type', ImageSequenceType.SEQUENTIAL),
            ('is_depth_available', True),
            ('is_normals_available', False),
            ('is_stereo_available', False),
            ('is_labels_available', False),
            ('is_masks_available', True),
            ('camera_intrinsics', images[0].metadata.intrinsics)
        ]
        for idx in range(len(arguments) + 1):
            collection = ic.ImageCollection(
                *(v for _, v in arguments[:idx]),
                **{k: v for k, v in arguments[idx:]}
            )

            # Check it got assigned correctly
            for name, value in arguments:
                self.assertEqual(value, getattr(collection, name))

    def test_infers_properties_from_mono_images(self):
        images = [make_image(idx, depth=None) for idx in range(10)]

        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.1 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        self.assertFalse(collection.is_depth_available)
        self.assertTrue(collection.is_normals_available)
        self.assertFalse(collection.is_stereo_available)
        self.assertTrue(collection.is_labels_available)
        self.assertFalse(collection.is_masks_available)

        self.assertEqual(collection.camera_intrinsics, images[0].metadata.intrinsics)
        self.assertIsNone(collection.right_camera_pose)
        self.assertIsNone(collection.right_camera_intrinsics)

        images = [
            make_image(idx, normals=None, metadata=make_metadata(
                idx,
                labelled_objects=[imeta.MaskedObject(
                    ('class_1',),
                    152,
                    239,
                    14,
                    78,
                    tf.Transform(location=(123, -45, 23), rotation=(0.5, 0.23, 0.1)),
                    'LabelledObject-18569',
                    np.random.choice((True, False), size=(78, 14)))]
            ))
            for idx in range(10)
        ]

        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.1 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        self.assertTrue(collection.is_depth_available)
        self.assertFalse(collection.is_normals_available)
        self.assertFalse(collection.is_stereo_available)
        self.assertTrue(collection.is_labels_available)
        self.assertTrue(collection.is_masks_available)

        self.assertEqual(collection.camera_intrinsics, images[0].metadata.intrinsics)
        self.assertIsNone(collection.right_camera_pose)
        self.assertIsNone(collection.right_camera_intrinsics)

    def test_infers_properties_from_stereo_images(self):
        images = [make_stereo_image(idx, depth=None) for idx in range(10)]

        collection = ic.ImageCollection(
            images=images,
            timestamps=[1.1 * idx for idx in range(10)],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        self.assertFalse(collection.is_depth_available)
        self.assertTrue(collection.is_normals_available)
        self.assertTrue(collection.is_stereo_available)
        self.assertTrue(collection.is_labels_available)
        self.assertFalse(collection.is_masks_available)

        self.assertEqual(collection.camera_intrinsics, images[0].metadata.intrinsics)
        self.assertEqual(collection.right_camera_pose,
                         images[0].left_camera_pose.find_relative(images[0].right_camera_pose))
        self.assertEqual(collection.right_camera_intrinsics, images[0].right_metadata.intrinsics)

    def test_iterates_over_timestamps_and_images_in_timestamp_order(self):
        pairs = [(10 - 0.8 * idx, make_image(idx, depth=None)) for idx in range(10)]
        collection = ic.ImageCollection(
            images=[pair[1] for pair in pairs],
            timestamps=[pair[0] for pair in pairs],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        pairs = sorted(pairs, key=itemgetter(0))
        prev_timestamp = None
        for idx, (timestamp, image) in enumerate(collection):
            self.assertEqual(pairs[idx][0], timestamp)
            self.assertEqual(pairs[idx][1], image)
            if prev_timestamp is not None:
                self.assertGreater(timestamp, prev_timestamp)
            prev_timestamp = timestamp

    # @mock.patch('arvet.core.image_collection.os.path.isfile', autospec=os.path.isfile)
    # def test_loads_images_from_cache_if_available(self, mock_isfile):
    #     db_client = self.create_mock_db_client()
    #     subject = self.make_instance(db_client_=db_client)
    #
    #     mock_image = make_image()
    #     mock_isfile.return_value = True
    #     mock_open = mock.mock_open(read_data=pickle.dumps(mock_image, protocol=pickle.HIGHEST_PROTOCOL))
    #
    #     with mock.patch('arvet.core.image_collection.open', mock_open, create=True):
    #         with subject:
    #             while not subject.is_complete():
    #                 image, stamp = subject.get_next_image()
    #                 self.assertTrue(mock_open.called)
    #                 self.assert_images_equal(mock_image, image)
    #                 # Check that we didn't ask the database
    #                 self.assertNotIn(mock.call({'_id': image.identifier}),
    #                                  db_client.image_collection.find_one.call_args_list)
    #
    # @mock.patch('arvet.core.image_collection.os.path.isfile', autospec=os.path.isfile)
    # def test_loads_images_on_the_fly(self, mock_isfile):
    #     db_client = self.create_mock_db_client()
    #     subject = self.make_instance(db_client_=db_client)
    #
    #     mock_isfile.return_value = False
    #     with subject:
    #         while not subject.is_complete():
    #             image, stamp = subject.get_next_image()
    #             # Check that we loaded the image
    #             self.assertIn(mock.call({'_id': image.identifier}), db_client.image_collection.find_one.call_args_list)
    #
    # def test_create_and_save_checks_image_ids_and_stops_if_not_all_found(self):
    #     db_client = self.create_mock_db_client()
    #     mock_cursor = mock.MagicMock()
    #     mock_cursor.count.return_value = len(self.image_map) - 2    # Return missing ids.
    #     db_client.image_collection.find.return_value = mock_cursor
    #
    #     result = ic.ImageCollection.create_and_save(
    #         db_client, {timestamp: image_id for timestamp, image_id in self.images.items()},
    #         arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
    #     self.assertIsNone(result)
    #     self.assertTrue(db_client.image_collection.find.called)
    #     self.assertEqual({'_id': {'$in': [image_id for image_id in self.images.values()]}},
    #                      db_client.image_collection.find.call_args[0][0])
    #     self.assertFalse(db_client.image_source_collection.insert.called)
    #
    # def test_create_and_save_checks_for_existing_collection(self):
    #     db_client = self.create_mock_db_client()
    #     ic.ImageCollection.create_and_save(db_client,
    #                                        {timestamp: image_id for timestamp, image_id in self.images.items()},
    #                                        arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
    #
    #     self.assertTrue(db_client.image_source_collection.find_one.called)
    #     existing_query = db_client.image_source_collection.find_one.call_args[0][0]
    #     self.assertIn('_type', existing_query)
    #     self.assertEqual('arvet.core.image_collection.ImageCollection', existing_query['_type'])
    #     self.assertIn('sequence_type', existing_query)
    #     self.assertEqual('SEQ', existing_query['sequence_type'])
    #     self.assertIn('images', existing_query)
    #     self.assertIn('$all', existing_query['images'])
    #     # Because dicts, we can't guarantee the order of this list
    #     # So we use $all, and make sure all the timestamp->image_id pairs are in it
    #     for timestamp, image_id in self.images.items():
    #         self.assertIn([timestamp, image_id], existing_query['images']['$all'])
    #
    # def test_create_and_save_makes_valid_collection(self):
    #     db_client = self.create_mock_db_client()
    #     mock_cursor = mock.MagicMock()
    #     mock_cursor.count.return_value = len(self.image_map)
    #     db_client.image_collection.find.return_value = mock_cursor
    #     db_client.image_source_collection.find_one.return_value = None
    #
    #     ic.ImageCollection.create_and_save(db_client,
    #                                        {timestamp: image_id for timestamp, image_id in self.images.items()},
    #                                        arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
    #     self.assertTrue(db_client.image_source_collection.insert_one.called)
    #     s_image_collection = db_client.image_source_collection.insert_one.call_args[0][0]
    #
    #     db_client = self.create_mock_db_client()
    #     collection = ic.ImageCollection.deserialize(s_image_collection, db_client)
    #     self.assertEqual(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL, collection.sequence_type)
    #     self.assertEqual(len(self.images), len(collection))
    #     for stamp, image_id in self.images.items():
    #         image = self.image_map[image_id]
    #         self.assertEqual(image.identifier, collection[stamp].identifier)
    #         self.assertTrue(np.array_equal(image.data, collection[stamp].data))
    #         self.assertEqual(image.camera_pose, collection[stamp].camera_pose)
    #         self.assertTrue(np.array_equal(image.depth_data, collection[stamp].depth_data))
    #         self.assertTrue(np.array_equal(image.labels_data, collection[stamp].labels_data))
    #         self.assertTrue(np.array_equal(image.world_normals_data, collection[stamp].world_normals_data))
    #         self.assertEqual(image.additional_metadata, collection[stamp].additional_metadata)
    #
    #     s_image_collection_2 = collection.serialize()
    #     # Pymongo converts tuples to lists
    #     s_image_collection_2['images'] =[list(elem) for elem in s_image_collection_2['images']]
    #     self.assert_serialized_equal(s_image_collection, s_image_collection_2)
    #
    # @mock.patch('arvet.core.image_collection.os.makedirs', autospec=os.makedirs)
    # def test_warmup_cache_creates_image_files(self, mock_makedirs):
    #     db_client = self.create_mock_db_client()
    #     subject = self.make_instance(db_client_=db_client)
    #
    #     mock_open = mock.mock_open()
    #     with mock.patch('arvet.core.image_collection.open', mock_open, create=True):
    #         subject.warmup_cache()
    #     self.assertEqual(mock.call('{0}/image_cache'.format(db_client.temp_folder), exist_ok=True),
    #                      mock_makedirs.call_args)
    #     for image_id in subject._images.values():
    #         self.assertIn(mock.call('{0}/image_cache/{1}.pickle'.format(db_client.temp_folder, image_id), 'wb'),
    #                       mock_open.call_args_list)


def make_metadata(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'img_hash': b'\xf1\x9a\xe2|' + np.random.randint(0, 0xFFFFFFFF).to_bytes(4, 'big'),
        'source_type': imeta.ImageSourceType.SYNTHETIC,
        'camera_pose': tf.Transform(location=(1 + 100 * index, 2 + np.random.uniform(-1, 1), 3),
                                    rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
        'intrinsics': cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
        'environment_type': imeta.EnvironmentType.INDOOR_CLOSE,
        'light_level': imeta.LightingLevel.WELL_LIT, 'time_of_day': imeta.TimeOfDay.DAY,
        'lens_focal_distance': 5,
        'aperture': 22,
        'simulation_world': 'TestSimulationWorld',
        'lighting_model': imeta.LightingModel.LIT,
        'texture_mipmap_bias': 1,
        'normal_maps_enabled': 2,
        'roughness_enabled': True,
        'geometry_decimation': 0.8,
        'procedural_generation_seed': 16234,
        'labelled_objects': (
            imeta.LabelledObject(
                class_names=('car',), x=12, y=144, width=67, height=43,
                relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)),
                instance_name='Car-002'
            ),
            imeta.LabelledObject(
                class_names=('cat',), x=125, y=244, width=117, height=67,
                relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                instance_name='cat-090'
            )
        )
    })
    return imeta.ImageMetadata(**kwargs)


def make_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'pixels': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': make_metadata(index),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth': np.random.uniform(0, 1, (32, 32)),
        'normals': np.random.uniform(0, 1, (32, 32, 3))
    })
    return im.Image(**kwargs)


def make_stereo_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'pixels': np.random.uniform(0, 255, (32, 32, 3)),
        'right_pixels': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': make_metadata(index),
        'right_metadata': make_metadata(
            index,
            camera_pose=tf.Transform(location=(1 + 100*index, 2 + np.random.uniform(-1, 1), 4),
                                     rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
        ),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth': np.random.uniform(0, 1, (32, 32)),
        'right_depth': np.random.uniform(0, 1, (32, 32)),
        'normals': np.random.uniform(0, 1, (32, 32, 3)),
        'right_normals': np.random.uniform(0, 1, (32, 32, 3))
    })
    return im.StereoImage(**kwargs)
