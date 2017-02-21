from unittest import TestCase
from numpy import array as nparray, array_equal
from core.image import Image


class TestImage(TestCase):

    def test_identifier(self):
        image = Image(1,'/home/user/test/', 13, 13/30, nparray([1, 2, 3]), nparray([1, 2, 3, 4]), id_=123)
        self.assertEquals(image.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a':1, 'b':2, 'c': 3}
        image = Image(1,'/home/user/test/', 13, 13/30, nparray([1, 2, 3]), nparray([1, 2, 3, 4]), **kwargs)
        self.assertEquals(image.identifier, 1234)

    def test_serialize_and_deserialize(self):
        image1 = Image(1,'/home/user/test/', 13, 13/30, nparray([1, 2, 3]), nparray([1, 2, 3, 4]), id_=12345)
        s_image1 = image1.serialize()

        image2 = Image.deserialize(s_image1)
        s_image2 = image2.serialize()

        self._assert_models_equal(image1, image2)
        self.assertEquals(s_image1, s_image2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            image2 = Image.deserialize(s_image2)
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
        if not isinstance(image1, Image) or not isinstance(image2, Image):
            self.fail('object was not an Image')
        self.assertEquals(image1.identifier, image2.identifier)
        self.assertEquals(image1.dataset, image2.dataset)
        self.assertEquals(image1.filename, image2.filename)
        self.assertEquals(image1.metadata_filename, image2.metadata_filename)
        self.assertEquals(image1.index, image2.index)
        self.assertEquals(image1.timestamp, image2.timestamp)
        self.assertTrue(array_equal(image1.camera_location, image2.camera_location))
        self.assertTrue(array_equal(image1.camera_orientation, image2.camera_orientation))
