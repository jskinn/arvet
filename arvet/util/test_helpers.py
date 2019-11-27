import numpy as np
import unittest


class ExtendedTestCase(unittest.TestCase):
    """
    Some utility extensions to the standard unittest test case, for working with numpy arrays.
    """

    def assertNPEqual(self, array_1, array_2, msg=''):
        self.assertTrue(np.array_equal(array_1, array_2),
                        "{0}: Arrays {1} and {2} are not equal".format(msg, str(array_1), str(array_2)))

    def assertNotNPEqual(self, array_1, array_2, msg=''):
        self.assertFalse(np.array_equal(array_1, array_2),
                         "{0}: Arrays {1} and {2} are equal".format(msg, str(array_1), str(array_2)))

    def assertNPClose(self, array_1, array_2, msg='', rtol=1e-05, atol=1e-08):
        self.assertTrue(np.all(np.isclose(array_1, array_2, rtol=rtol, atol=atol)),
                        "{0}: Arrays {1} and {2} are not close".format(msg, str(array_1), str(array_2)))

    def assertNotNPClose(self, array_1, array_2, msg='', rtol=1e-05, atol=1e-08):
        self.assertFalse(np.all(np.isclose(array_1, array_2, rtol=rtol, atol=atol)),
                         "{0}: Arrays {1} and {2} are close together".format(msg, str(array_1), str(array_2)))

