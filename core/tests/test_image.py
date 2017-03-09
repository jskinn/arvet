import unittest
import numpy as np
import core.image as im


class TestImage(unittest.TestCase):

    def setUp(self):
        self.image = im.Image(13/30,'/home/user/test.png', np.array([1, 2, 3]), np.array([4, 5, 6, 7]))

    def test_filename(self):
        self.assertEquals(self.image.filename, '/home/user/test.png')

    def test_timestamp(self):
        self.assertEquals(self.image.timestamp, 13/30)

    def test_camera_location(self):
        self.assertTrue(np.array_equal(self.image.camera_location, np.array([1, 2, 3])))

    def test_padded_kwargs(self):
        kwargs = {'a':1, 'b':2, 'c': 3}
        with self.assertRaises(TypeError):
            im.Image(13/30,'/home/user/test.png', np.array([1, 2, 3]), np.array([1, 2, 3, 4]), **kwargs)

class TestStereoImage(unittest.TestCase):

    def setUp(self):
        self.image = im.StereoImage(timestamp=13/30,
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
            #Extras:
            'a': 1, 'b': 2, 'c': 3
        }
        with self.assertRaises(TypeError):
            im.StereoImage(**kwargs)

    def test_left_image_is_base(self):
        self.assertEquals(self.image.filename, self.image.left_filename)
        self.assertTrue(np.array_equal(self.image.camera_location, self.image.left_camera_location))
        self.assertTrue(np.array_equal(self.image.camera_orientation, self.image.left_camera_orientation))
