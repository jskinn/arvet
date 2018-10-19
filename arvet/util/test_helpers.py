import numpy as np
import unittest


class ExtendedTestCase(unittest.TestCase):
    """
    Some utility extensions to the standard unittest test case, for working with numpy arrays.
    """

    def assertNPEqual(self, array_1, array_2):
        self.assertTrue(np.array_equal(array_1, array_2),
                        "Arrays {0} and {1} are not equal".format(str(array_1), str(array_2)))

    def assertNPClose(self, array_1, array_2):
        self.assertTrue(np.all(np.isclose(array_1, array_2)),
                        "Arrays {0} and {1} are not close".format(str(array_1), str(array_2)))

