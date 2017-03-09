import unittest
import numpy as np
import util.dict_utils as du
import database.tests.test_entity as entity_test
import core.image_entity as ie


class TestImageEntity(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return ie.ImageEntity

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'timestamp': 13 / 30,
            'filename': '/home/user/test.png',
            'camera_location': np.array([1, 2, 3]),
            'camera_orientation': np.array([4, 5, 6, 7]),
            'additional_metadata': {
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            },
            'depth_filename': '/home/user/test_depth.png',
            'labels_filename': '/home/user/test_labels.png',
            'world_normals_filename': '/home/user/test_normals.png'
        })
        return ie.ImageEntity(*args, **kwargs)

    def assert_models_equal(self, image1, image2):
        """
        Helper to assert that two dataset models are equal
        :param image1: Dataset
        :param image2: Dataset
        :return:
        """
        if not isinstance(image1, ie.ImageEntity) or not isinstance(image2, ie.ImageEntity):
            self.fail('object was not an Image')
        self.assertEquals(image1.identifier, image2.identifier)
        self.assertEquals(image1.filename, image2.filename)
        self.assertEquals(image1.timestamp, image2.timestamp)
        self.assertTrue(np.array_equal(image1.camera_location, image2.camera_location))
        self.assertTrue(np.array_equal(image1.camera_orientation, image2.camera_orientation))
        self.assertEquals(image1.depth_filename, image2.depth_filename)
        self.assertEquals(image1.labels_filename, image2.labels_filename)
        self.assertEquals(image1.world_normals_filename, image2.world_normals_filename)
        self.assertEquals(image1.additional_metadata, image2.additional_metadata)


class TestStereoImageEntity(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return ie.StereoImageEntity

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'timestamp': 13 / 30,
            'left_filename': '/home/user/left.png',
            'left_camera_location': np.array([1, 2, 3]),
            'left_camera_orientation': np.array([4, 5, 6, 7]),
            'left_depth_filename': '/home/user/left_depth.png',
            'left_labels_filename': '/home/user/left_labels.png',
            'left_world_normals_filename': '/home/user/left_normals.png',
            'right_filename': '/home/user/right.png',
            'right_camera_location': np.array([8, 9, 10]),
            'right_camera_orientation': np.array([11, 12, 13, 14]),
            'right_depth_filename': '/home/user/right_depth.png',
            'right_labels_filename': '/home/user/right_labels.png',
            'right_world_normals_filename': '/home/user/right_normals.png',
            'additional_metadata': {
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        })
        return ie.StereoImageEntity(*args, **kwargs)

    def assert_models_equal(self, image1, image2):
        """
        Helper to assert that two dataset models are equal
        :param image1: Dataset
        :param image2: Dataset
        :return:
        """
        if not isinstance(image1, ie.StereoImageEntity) or not isinstance(image2, ie.StereoImageEntity):
            self.fail('object was not an Image')
        self.assertEquals(image1.identifier, image2.identifier)
        self.assertEquals(image1.timestamp, image2.timestamp)
        self.assertEquals(image1.left_filename, image2.left_filename)
        self.assertEquals(image1.right_filename, image2.right_filename)
        self.assertTrue(np.array_equal(image1.left_camera_location, image2.left_camera_location))
        self.assertTrue(np.array_equal(image1.left_camera_orientation, image2.left_camera_orientation))
        self.assertTrue(np.array_equal(image1.right_camera_location, image2.right_camera_location))
        self.assertTrue(np.array_equal(image1.right_camera_orientation, image2.right_camera_orientation))
        self.assertEquals(image1.left_depth_filename, image2.left_depth_filename)
        self.assertEquals(image1.left_labels_filename, image2.left_labels_filename)
        self.assertEquals(image1.left_world_normals_filename, image2.left_world_normals_filename)
        self.assertEquals(image1.right_depth_filename, image2.right_depth_filename)
        self.assertEquals(image1.right_labels_filename, image2.right_labels_filename)
        self.assertEquals(image1.right_world_normals_filename, image2.right_world_normals_filename)
        self.assertEquals(image1.additional_metadata, image2.additional_metadata)
