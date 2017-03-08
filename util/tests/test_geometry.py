import unittest
import numpy as np
import util.geometry as g


class TestGeometry(unittest.TestCase):

    def test_dict_vector_to_np_array(self):
        result = g.dict_vector_to_np_array({'X': 1, 'Y': 2, 'Z': 3})
        self.assertIsInstance(result, np.ndarray)
        self.assertEquals(result.shape, (3,))
        self.assertEquals(result[0], 1)
        self.assertEquals(result[1], 2)
        self.assertEquals(result[2], 3)

    def test_dict_vector_to_np_array_causes_exception_for_missing_keys(self):
        got_error = False
        try:
            g.dict_vector_to_np_array({'x': 1, 'Y': 2, 'Z': 3})
        except KeyError:
            got_error = True
        self.assertTrue(got_error)

    def test_dict_quaternion_to_np_array(self):
        result = g.dict_quaternion_to_np_array({'X': 1, 'Y': 2, 'Z': 3, 'W': 4})
        self.assertIsInstance(result, np.ndarray)
        self.assertEquals(result.shape, (4,))
        self.assertEquals(result[0], 1)
        self.assertEquals(result[1], 2)
        self.assertEquals(result[2], 3)
        self.assertEquals(result[3], 4)

    def test_numpy_vector_to_dict(self):
        result = g.numpy_vector_to_dict(np.array([1,2,3]))
        self.assertEquals(result, {'X': 1, 'Y': 2, 'Z': 3})

    def test_numpy_quaternion_to_dict(self):
        result = g.numpy_quarternion_to_dict(np.array([1, 2, 3, 4]))
        self.assertEquals(result, {'X': 1, 'Y': 2, 'Z': 3, 'W': 4})

    def test_square_length(self):
        result = g.square_length(np.array([1,2,3]))
        self.assertEquals(result, 14)

    def test_square_length_detailed(self):
        for a in range(-10, 10, 3):
            x = a * 1.63436326 * np.sin(a)
            y = x * 8.46546780 * np.sin(x)
            z = y * 4.27345834 * np.sin(y)
            print("{0}, {1}, {2}".format(x,y,z))
            self.assertEquals(g.square_length(np.array([x, y, z])), x*x + y*y + z*z)
