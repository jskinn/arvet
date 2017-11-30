# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import pymongo
import bson
import gridfs
import argus.database.client
import argus.core.image_entity
import argus.core.image_source
import argus.core.sequence_type
import argus.util.transform as tf
import argus.util.dict_utils as du
import argus.metadata.image_metadata as imeta
import argus.image_collections.image_collection_builder as image_collection_builder


class MockImageSource(argus.core.image_source.ImageSource):
    sequence_type = argus.core.sequence_type.ImageSequenceType.SEQUENTIAL

    def __init__(self, images):
        self.images = images
        self.index = 0

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def begin(self):
        self.index = 0

    def is_complete(self):
        return self.index >= len(self.images)

    def get_next_image(self):
        image = self.images[self.index]
        timestamp = self.index / 30
        self.index += 1
        return image, timestamp

    @property
    def is_stored_in_database(self):
        return False

    @property
    def is_labels_available(self):
        return False

    @property
    def is_normals_available(self):
        return False

    def get(self, index):
        return None

    def get_camera_intrinsics(self):
        return None

    @property
    def is_per_pixel_labels_available(self):
        return False

    @property
    def is_depth_available(self):
        return False

    @property
    def is_stereo_available(self):
        return False

    @property
    def supports_random_access(self):
        return False


def make_image(*args, **kwargs):
    du.defaults(kwargs, {
        'data': np.random.randint(0, 255, (32, 32, 3), dtype='uint8'),
        'data_id': 0,
        'metadata': imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            hash_=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            camera_pose=tf.Transform()
        )
    })
    return argus.core.image_entity.ImageEntity(*args, **kwargs)


class TestImageCollectionBuilder(unittest.TestCase):

    def setUp(self):
        self.mock_db_client = mock.create_autospec(argus.database.client.DatabaseClient, spec_set=True)
        self.mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection, spec_set=True)
        self.mock_db_client.image_collection.find.return_value = mock.create_autospec(pymongo.cursor.Cursor,
                                                                                      spec_set=True)
        self.mock_db_client.image_collection.find_one.return_value = None
        self.mock_db_client.image_collection.insert.return_value = bson.objectid.ObjectId()
        self.mock_db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection, spec_set=True)
        self.mock_db_client.image_source_collection.find_one.return_value = None
        self.mock_db_client.image_source_collection.insert.return_value = bson.objectid.ObjectId()
        self.mock_db_client.grid_fs = mock.create_autospec(gridfs.GridFS)

    def test_add_image_saves_image_to_database(self):
        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        image = make_image()
        subject.add_image(image)
        self.assertTrue(self.mock_db_client.image_collection.insert.called)
        self.assertIn(mock.call(image.serialize()), self.mock_db_client.image_collection.insert.mock_calls)

    def test_add_image_does_not_save_image_if_has_identifier(self):
        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_image(make_image(id_=bson.ObjectId()))
        self.assertFalse(self.mock_db_client.image_collection.insert.called)

    def test_add_image_assigns_timestamp(self):
        self.mock_db_client.image_collection.find.return_value.count.return_value = 1
        timestamp = np.random.uniform(0, 100)
        img_id = bson.ObjectId()
        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_image(make_image(id_=img_id), timestamp)
        subject.save()  # Save so we can extract the built collection
        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        self.assertEqual([(timestamp, img_id)],
                         self.mock_db_client.image_source_collection.insert.call_args[0][0]['images'])

    def test_add_image_auto_assigns_timestamp(self):
        img_ids = [bson.ObjectId for _ in range(4)]
        self.mock_db_client.image_collection.find.return_value.count.return_value = len(img_ids)
        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_image(make_image(id_=img_ids[0]))
        subject.add_image(make_image(id_=img_ids[1]))
        subject.add_image(make_image(id_=img_ids[2]), 2.2)
        subject.add_image(make_image(id_=img_ids[3]))
        subject.save()  # Save so we can extract the built collection
        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        s_images_list = self.mock_db_client.image_source_collection.insert.call_args[0][0]['images']
        self.assertEqual(4, len(s_images_list))
        self.assertIn((0, img_ids[0]), s_images_list)
        self.assertIn((1, img_ids[1]), s_images_list)
        self.assertIn((2.2, img_ids[2]), s_images_list)
        self.assertIn((3.2, img_ids[3]), s_images_list)

    def test_add_from_image_source_loops_over_image_source(self):
        inner_image_source = MockImageSource([make_image()])
        mock_image_source = mock.Mock(wraps=inner_image_source, spec_set=inner_image_source)
        mock_image_source.__enter__ = mock.Mock(wraps=inner_image_source.__enter__)
        mock_image_source.__exit__ = mock.Mock(wraps=inner_image_source.__exit__)

        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        self.assertEqual('__enter__', mock_image_source.mock_calls[0][0])
        self.assertEqual('is_complete', mock_image_source.mock_calls[1][0])
        self.assertEqual('get_next_image', mock_image_source.mock_calls[2][0])
        self.assertEqual('is_complete', mock_image_source.mock_calls[3][0])
        self.assertEqual('__exit__', mock_image_source.mock_calls[4][0])

    def test_add_from_image_source_can_offset_timestamps(self):
        img_ids = [bson.ObjectId for _ in range(2)]
        self.mock_db_client.image_collection.find.return_value.count.return_value = 4
        mock_image_source = MockImageSource([make_image(id_=img_id) for img_id in img_ids])

        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.add_from_image_source(mock_image_source, offset=600)
        subject.save()  # Save so we can extract the built collection
        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        s_images_list = self.mock_db_client.image_source_collection.insert.call_args[0][0]['images']
        self.assertEqual(4, len(s_images_list))
        self.assertIn((0, img_ids[0]), s_images_list)
        self.assertIn((1/30, img_ids[1]), s_images_list)
        self.assertIn((600, img_ids[0]), s_images_list)
        self.assertIn((600 + 1/30, img_ids[1]), s_images_list)

    def test_save_checks_for_existing(self):
        ids = [bson.ObjectId() for _ in range(4)]
        self.mock_db_client.image_collection.find.return_value.count.return_value = 4
        self.mock_db_client.image_collection.insert.side_effect = ids
        self.mock_db_client.image_source_collection.find_one.return_value = {'_id': bson.ObjectId()}
        mock_image_source = MockImageSource([make_image() for _ in range(4)])

        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.save()

        self.assertTrue(self.mock_db_client.image_source_collection.find_one.called)
        query = self.mock_db_client.image_source_collection.find_one.call_args[0][0]
        # Evaluate the query in bits, because we can't guarantee the order of the images
        self.assertEqual({'_type', 'images', 'sequence_type'}, set(query.keys()))
        self.assertEqual('argus.core.image_collection.ImageCollection', query['_type'])
        self.assertEqual('SEQ', query['sequence_type'])
        for idx, img_id in enumerate(ids):
            self.assertIn((idx / 30, img_id), query['images']['$all'])
        self.assertFalse(self.mock_db_client.image_source_collection.insert.called)

    def test_save_stores_image_collection(self):
        ids = [bson.ObjectId() for _ in range(4)]
        self.mock_db_client.image_collection.find.return_value.count.return_value = 4
        self.mock_db_client.image_collection.insert.side_effect = ids
        mock_image_source = MockImageSource([make_image() for _ in range(4)])

        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.save()

        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        query = self.mock_db_client.image_source_collection.insert.call_args[0][0]
        # Evaluate the query in bits, because we can't guarantee the order of the images
        self.assertEqual({'_type', 'images', 'sequence_type'}, set(query.keys()))
        self.assertEqual('argus.core.image_collection.ImageCollection', query['_type'])
        self.assertEqual('SEQ', query['sequence_type'])
        for idx, img_id in enumerate(ids):
            self.assertIn((idx / 30, img_id), query['images'])

    def test_adding_from_non_sequential_source_makes_sequence_non_sequential(self):
        self.mock_db_client.image_collection.find.return_value.count.return_value = 4
        self.mock_db_client.image_collection.insert.side_effect = [bson.ObjectId() for _ in range(4)]
        mock_image_source = MockImageSource([make_image() for _ in range(4)])
        mock_image_source.sequence_type = argus.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.save()
        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        self.assertEqual('NON', self.mock_db_client.image_source_collection.insert.call_args[0][0]['sequence_type'])

    def test_adding_from_multiple_sources_makes_sequence_non_sequential(self):
        self.mock_db_client.image_collection.find.return_value.count.return_value = 8
        self.mock_db_client.image_collection.insert.side_effect = [bson.ObjectId() for _ in range(8)]
        mock_image_source = MockImageSource([make_image() for _ in range(4)])

        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.add_from_image_source(mock_image_source, offset=100)
        subject.save()
        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        self.assertEqual('NON', self.mock_db_client.image_source_collection.insert.call_args[0][0]['sequence_type'])

    def test_set_non_sequential(self):
        self.mock_db_client.image_collection.find.return_value.count.return_value = 1
        subject = image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_image(make_image())
        subject.set_non_sequential()
        subject.save()
        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        self.assertEqual('NON', self.mock_db_client.image_source_collection.insert.call_args[0][0]['sequence_type'])
