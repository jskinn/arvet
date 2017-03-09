import unittest
import numpy as np
import core.image_entity as ie


class TestImageEntity(unittest.TestCase):

    def setUp(self):
        self.image = ie.ImageEntity(13 / 30, '/home/user/test/', np.array([1, 2, 3]), np.array([1, 2, 3, 4]))

    def test_padded_kwargs(self):
        kwargs = {'a': 1, 'b': 2, 'c': 3}
        with self.assertRaises(TypeError):
            ie.ImageEntity(13 / 30, '/home/user/test/', np.array([1, 2, 3]), np.array([1, 2, 3, 4]), **kwargs)

    def test_serialize_and_deserialize(self):
        image1 = ie.ImageEntity(13 / 30, '/home/user/test/', np.array([1, 2, 3]), np.array([1, 2, 3, 4]), id_=12345)
        s_image1 = image1.serialize()

        image2 = ie.ImageEntity.deserialize(s_image1)
        s_image2 = image2.serialize()

        self._assert_models_equal(image1, image2)
        self.assertEquals(s_image1, s_image2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            image2 = ie.ImageEntity.deserialize(s_image2)
            s_image2 = image2.serialize()
            self._assert_models_equal(image1, image2)
            self.assertEquals(s_image1, s_image2)

    def _assert_models_equal(self, image1, image2):
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


class TestStereoImageEntity(unittest.TestCase):

    def setUp(self):
        self.image = ie.StereoImageEntity(timestamp=13 / 30,
                                          left_filename='/home/user/left.png',
                                          left_camera_location=np.array([1, 2, 3]),
                                          left_camera_orientation=np.array([4, 5, 6, 7]),
                                          right_filename='/home/user/right.png',
                                          right_camera_location=np.array([8, 9, 10]),
                                          right_camera_orientation=np.array([11, 12, 13, 14]))

    def test_padded_kwargs(self):
        kwargs = {
            'timestamp': 13 / 30,
            'left_filename': '/home/user/left.png',
            'left_camera_location': np.array([1, 2, 3]),
            'left_camera_orientation': np.array([4, 5, 6, 7]),
            'right_filename': '/home/user/right.png',
            'right_camera_location': np.array([8, 9, 10]),
            'right_camera_orientation': np.array([11, 12, 13, 14]),
            # Extras:
            'a': 1, 'b': 2, 'c': 3
        }
        with self.assertRaises(TypeError):
            ie.StereoImageEntity(**kwargs)

    def test_serialize_and_deserialize(self):
        image1 = ie.ImageEntity(13 / 30, '/home/user/test/', np.array([1, 2, 3]), np.array([1, 2, 3, 4]), id_=12345)
        s_image1 = image1.serialize()

        image2 = ie.ImageEntity.deserialize(s_image1)
        s_image2 = image2.serialize()

        self._assert_models_equal(image1, image2)
        self.assertEquals(s_image1, s_image2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            image2 = ie.ImageEntity.deserialize(s_image2)
            s_image2 = image2.serialize()
            self._assert_models_equal(image1, image2)
            self.assertEquals(s_image1, s_image2)

    def _assert_models_equal(self, image1, image2):
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
