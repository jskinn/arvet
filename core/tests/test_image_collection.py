import unittest
import unittest.mock as mock
import database.tests.test_entity
import numpy as np
import pymongo.collection
import bson.objectid
import util.dict_utils as du
import util.transform as tf
import core.image_entity as ie
import core.image_collection as ic
import core.sequence_type


def make_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'id_': bson.objectid.ObjectId(),
        'timestamp': 127 / 31 * index + np.random.uniform(-0.1, 0.1),
        'data': np.random.uniform(0, 255, (32, 32, 3)),
        'camera_pose': tf.Transform(location=(1 + 100*index, 2 + np.random.uniform(-1, 1), 3),
                                    rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
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


class TestImageCollection(database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.image_map = {}
        self.images_list = []
        for i in range(10):
            image = make_image(i)
            self.image_map[str(image.identifier)] = image
            self.images_list.append(image)

    def get_class(self):
        return ic.ImageCollection

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'images': self.images_list,
            'type_': core.sequence_type.ImageSequenceType.SEQUENTIAL
        })
        return ic.ImageCollection(*args, **kwargs)

    def assert_models_equal(self, image_collection1, image_collection2):
        """
        Helper to assert that two image entities are equal
        :param image_collection1: ImageEntity
        :param image_collection2: ImageEntity
        :return:
        """
        if not isinstance(image_collection1, ic.ImageCollection) or not isinstance(image_collection2, ic.ImageCollection):
            self.fail('object was not an Image Collection')
        self.assertEqual(image_collection1.identifier, image_collection2.identifier)
        self.assertEqual(image_collection1.sequence_type, image_collection2.sequence_type)
        self.assertEqual(image_collection1.is_depth_available, image_collection2.is_depth_available)
        self.assertEqual(image_collection1.is_labels_available, image_collection2.is_labels_available)
        self.assertEqual(image_collection1.is_normals_available, image_collection2.is_normals_available)
        self.assertEqual(image_collection1.is_stereo_available, image_collection2.is_stereo_available)
        self.assertEqual(len(image_collection1), len(image_collection2))
        for idx in range(len(image_collection1)):
            self.assertEqual(image_collection1[idx].identifier, image_collection2[idx].identifier)
            self.assertEqual(image_collection1[idx].timestamp, image_collection2[idx].timestamp)
            self.assertTrue(np.array_equal(image_collection1[idx].data, image_collection2[idx].data))
            self.assertEqual(image_collection1[idx].camera_pose, image_collection2[idx].camera_pose)
            self.assertTrue(np.array_equal(image_collection1[idx].depth_data, image_collection2[idx].depth_data))
            self.assertTrue(np.array_equal(image_collection1[idx].labels_data, image_collection2[idx].labels_data))
            self.assertTrue(np.array_equal(image_collection1[idx].world_normals_data,
                                           image_collection2[idx].world_normals_data))
            self.assertEqual(image_collection1[idx].additional_metadata, image_collection2[idx].additional_metadata)

    def create_mock_db_client(self):
        self.db_client = super().create_mock_db_client()

        self.db_client.image_collection.find.return_value.sort.return_value = [image.serialize() for image in self.images_list]
        self.db_client.deserialize_entity.side_effect = lambda s_image: self.image_map[str(s_image['_id'])]

        return self.db_client

    def test_deserializes_images(self):
        s_image_collection = {
            '_id': 12345,
            'images': [image.identifier for image in self.images_list],
            '_type': 'ImageCollection',
            'sequence_type': 'SEQ'
        }
        db_client = self.create_mock_db_client()

        ic.ImageCollection.deserialize(s_image_collection, db_client)
        self.assertIn(mock.call({'_id': {'$in': s_image_collection['images']}}),
                      db_client.image_collection.find.call_args_list)
        for image in self.images_list:
            self.assertIn(mock.call(image.serialize()), db_client.deserialize_entity.call_args_list)
