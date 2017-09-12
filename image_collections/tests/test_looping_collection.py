#Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import pymongo.collection
import bson.objectid
import database.tests.test_entity
import metadata.image_metadata as imeta
import core.image
import core.sequence_type
import core.image_collection
import image_collections.looping_collection as looper
import util.dict_utils as du


class TestAugmentedCollection(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return looper.LoopingCollection

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'inner': make_image_collection(),
            'repeats': 3,
            'type_override': core.sequence_type.ImageSequenceType.SEQUENTIAL
        })
        return looper.LoopingCollection(*args, **kwargs)

    def assert_models_equal(self, collection1, collection2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param collection1:
        :param collection2:
        :return:
        """
        if (not isinstance(collection1, looper.LoopingCollection) or
                not isinstance(collection2, looper.LoopingCollection)):
            self.fail('object was not an LoopingCollection')
        self.assertEqual(collection1.identifier, collection2.identifier)
        self.assertEqual(collection1._inner.identifier, collection2._inner.identifier)
        self.assertEqual(collection1._repeats, collection2._repeats)
        self.assertEqual(collection1._type_override, collection2._type_override)

    def create_mock_db_client(self):
        db_client = super().create_mock_db_client()
        db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection)
        db_client.image_source_collection.find_one.side_effect = lambda q: {
            '_type': 'core.image_collection.ImageCollection',
            '_id': q['_id']
        }
        db_client.deserialize_entity.side_effect = mock_deserialize_entity
        return db_client

    def test_begin_calls_begin_on_inner(self):
        inner = mock.create_autospec(core.image_collection.ImageCollection)
        inner.get_next_image.return_value = (make_image(), 10)
        subject = looper.LoopingCollection(inner, 3)
        subject.begin()
        self.assertTrue(inner.begin.called)

    def test_returns_inner_images_in_order(self):
        img1 = make_image(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        img2 = make_image(data=np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]]))
        collection = make_image_collection(images={1: img1, 2: img2})
        subject = looper.LoopingCollection(collection, 3)

        subject.begin()
        for _ in range(3):
            img, _ = subject.get_next_image()
            self.assertNPEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9]], img.data)
            img, _ = subject.get_next_image()
            self.assertNPEqual([[11, 12, 13], [14, 15, 16], [17, 18, 19]], img.data)
        self.assertTrue(subject.is_complete())

    def test_singe_repeat_is_same_as_inner_collection(self):
        images = {idx + np.random.uniform(-0.2, 0.2):
                  make_image(data=np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)) for idx in range(10)}
        collection = make_image_collection(images=images)
        subject = looper.LoopingCollection(collection, 1)

        self.assertEqual(10, len(subject))
        img_count = 0
        prev_stamp = -1
        subject.begin()
        while not subject.is_complete():
            img, stamp = subject.get_next_image()
            self.assertIn(stamp, images)
            self.assertNPEqual(images[stamp].data, img.data)
            self.assertGreater(stamp, prev_stamp)
            prev_stamp = stamp
            img_count += 1
        self.assertEqual(img_count, 10)

    def test_stacks_stamp_on_subsequent_loops(self):
        images = {idx + np.random.uniform(-0.2, 0.2):
                  make_image(data=np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)) for idx in range(10)}
        collection = make_image_collection(images=images)
        subject = looper.LoopingCollection(collection, 2)
        timestamps = sorted(images.keys())
        max_timestamp = max(timestamps)

        self.assertEqual(20, len(subject))
        subject.begin()
        for idx in range(10):
            _, stamp = subject.get_next_image()
            self.assertEqual(timestamps[idx], stamp, "Failed on image {0}".format(idx))
        for idx in range(10):
            _, stamp = subject.get_next_image()
            self.assertEqual(timestamps[idx] + max_timestamp, stamp)
        self.assertTrue(subject.is_complete())

    def test_can_override_sequence_type(self):
        inner = mock.create_autospec(core.image_collection.ImageCollection)
        inner.get_next_image.return_value = (make_image(), 10)
        inner.sequence_type.return_value = core.sequence_type.ImageSequenceType.NON_SEQUENTIAL
        subject = looper.LoopingCollection(inner, 3, type_override=core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertEqual(core.sequence_type.ImageSequenceType.SEQUENTIAL, subject.sequence_type)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def make_image(**kwargs):
    if 'data' in kwargs:
        data = kwargs['data']
    else:
        data = np.array([list(range(i, i + 100)) for i in range(100)])
    metadata_kwargs = {
        'source_type': imeta.ImageSourceType.SYNTHETIC,
        'width': data.shape[1],
        'height': data.shape[0],
        'hash_': b'\xa5\xc9\x08\xaf$\x0b\x116'
    }
    if 'metadata' in kwargs and isinstance(kwargs['metadata'], dict):
        metadata_kwargs = du.defaults(kwargs['metadata'], metadata_kwargs)
        del kwargs['metadata']
    kwargs = du.defaults(kwargs, {
        'data': data,
        'metadata': imeta.ImageMetadata(**metadata_kwargs)
    })
    return core.image.Image(**kwargs)


def make_image_collection(**kwargs):
    if 'images' not in kwargs:
        kwargs['images'] = {1: make_image()}
    du.defaults(kwargs, {
        'type_': core.sequence_type.ImageSequenceType.SEQUENTIAL,
        'id_': bson.ObjectId()
    })
    return core.image_collection.ImageCollection(**kwargs)


def mock_deserialize_entity(s_entity):
    if s_entity['_type'] == 'core.image_collection.ImageCollection':
        return make_image_collection(id_=s_entity['_id'])
    return None
