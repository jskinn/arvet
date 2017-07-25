import unittest
import unittest.mock as mock
import numpy as np
import pymongo
import bson
import gridfs
import database.client
import core.image_entity
import core.image_source
import core.sequence_type
import util.transform as tf
import util.dict_utils as du
import metadata.image_metadata as imeta
import dataset.image_collection_builder


class MockImageSource:
    sequence_type = core.sequence_type.ImageSequenceType.SEQUENTIAL

    def __init__(self, images):
        self.images = images
        self.index = 0

    def begin(self):
        self.index = 0

    def is_complete(self):
        return self.index >= len(self.images)

    def get_next_image(self):
        image = self.images[self.index]
        self.index += 1
        return image, self.index / 30


def make_image(*args, **kwargs):
    du.defaults(kwargs, {
        'data': np.random.randint(0, 255, (32, 32, 3), dtype='uint8'),
        'data_id': 0,
        'metadata': imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            hash_=b'\x1f`\xa8\x8aR\xed\x9f\x0b', height=600, width=800,
            camera_pose=tf.Transform()
        )
    })
    return core.image_entity.ImageEntity(*args, **kwargs)


class TestImageCollectionBuilder(unittest.TestCase):

    def setUp(self):
        self.mock_db_client = mock.create_autospec(database.client.DatabaseClient, spec_set=True)
        self.mock_db_client.image_collection = mock.create_autospec(pymongo.collection.Collection, spec_set=True)
        self.mock_db_client.image_collection.find_one.return_value = None
        self.mock_db_client.image_collection.insert.return_value = bson.objectid.ObjectId()
        self.mock_db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection, spec_set=True)
        self.mock_db_client.image_source_collection.find_one.return_value = None
        self.mock_db_client.image_source_collection.insert.return_value = bson.objectid.ObjectId()
        self.mock_db_client.grid_fs = mock.create_autospec(gridfs.GridFS)

    def test_add_image_saves_image_to_database(self):
        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        image = make_image()
        subject.add_image(image)
        self.assertTrue(self.mock_db_client.image_collection.insert.called)
        self.assertIn(mock.call(image.serialize()), self.mock_db_client.image_collection.insert.mock_calls)

    def test_add_image_does_not_save_image_if_has_identifier(self):
        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_image(make_image(id_=bson.ObjectId()))
        self.assertFalse(self.mock_db_client.image_collection.insert.called)

    def test_add_from_image_source_loops_over_image_source(self):
        inner_image_source = MockImageSource([make_image()])
        mock_image_source = mock.Mock(wraps=inner_image_source, spec_set=inner_image_source)

        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        self.assertEqual('begin', mock_image_source.method_calls[0][0])
        self.assertEqual('is_complete', mock_image_source.method_calls[1][0])
        self.assertEqual('get_next_image', mock_image_source.method_calls[2][0])
        self.assertEqual('is_complete', mock_image_source.method_calls[3][0])

    def test_save_stores_image_collection(self):
        ids = [bson.ObjectId() for _ in range(4)]
        self.mock_db_client.image_collection.insert.side_effect = ids
        inner_image_source = MockImageSource([make_image() for _ in range(4)])
        mock_image_source = mock.Mock(wraps=inner_image_source, spec_set=inner_image_source)

        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.save()

        self.assertTrue(self.mock_db_client.image_source_collection.insert.called)
        self.assertEqual({
            '_type': 'ImageCollection',
            'images': ids,
            'sequence_type': 'SEQ'
        }, self.mock_db_client.image_source_collection.insert.call_args[0][0])

    def test_save_checks_for_existing(self):
        ids = [bson.ObjectId() for _ in range(4)]
        self.mock_db_client.image_collection.insert.side_effect = ids
        self.mock_db_client.image_source_collection.find_one.return_value = {'_id': bson.ObjectId()}
        inner_image_source = MockImageSource([make_image() for _ in range(4)])
        mock_image_source = mock.Mock(wraps=inner_image_source, spec_set=inner_image_source)

        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.save()

        self.assertTrue(self.mock_db_client.image_source_collection.find_one.called)
        self.assertEqual({
            '_type': 'ImageCollection',
            'images': ids,
            'sequence_type': 'SEQ'
        }, self.mock_db_client.image_source_collection.find_one.call_args[0][0])
        self.assertFalse(self.mock_db_client.image_source_collection.insert.called)

    def test_adding_from_non_sequential_source_makes_sequence_non_sequential(self):
        self.mock_db_client.image_collection.insert.side_effect = ids = [bson.ObjectId() for _ in range(4)]
        inner_image_source = MockImageSource([make_image() for _ in range(4)])
        mock_image_source = mock.Mock(wraps=inner_image_source, spec_set=inner_image_source)
        mock_image_source.sequence_type = core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.save()
        self.assertEqual('NON', self.mock_db_client.image_source_collection.insert.call_args[0][0]['sequence_type'])

    def test_adding_from_multiple_sources_makes_sequence_non_sequential(self):
        self.mock_db_client.image_collection.insert.side_effect = [bson.ObjectId() for _ in range(8)]
        inner_image_source = MockImageSource([make_image() for _ in range(4)])
        mock_image_source = mock.Mock(wraps=inner_image_source, spec_set=inner_image_source)

        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_from_image_source(mock_image_source)
        subject.add_from_image_source(mock_image_source)
        subject.save()
        self.assertEqual('NON', self.mock_db_client.image_source_collection.insert.call_args[0][0]['sequence_type'])

    def test_set_non_sequential(self):
        subject = dataset.image_collection_builder.ImageCollectionBuilder(self.mock_db_client)
        subject.add_image(make_image())
        subject.set_non_sequential()
        subject.save()
        self.assertEqual('NON', self.mock_db_client.image_source_collection.insert.call_args[0][0]['sequence_type'])
