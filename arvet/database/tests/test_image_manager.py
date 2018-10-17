import os.path
import unittest.mock
import numpy as np
import h5py
import arvet.database.image_manager as im_manager
from arvet.util.test_helpers import ExtendedTestCase


class TestDefaultImageManager(ExtendedTestCase):

    _tempfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-image-manager.h5py')

    @classmethod
    def setUpClass(cls):
        if os.path.isfile(cls._tempfile):
            os.remove(cls._tempfile)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls._tempfile):
            os.remove(cls._tempfile)

    def make_instance(self):
        return im_manager.DefaultImageManager(self._tempfile)

    def test_store_and_get_image(self):
        subject = self.make_instance()
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path = subject.store_image(image)
        output = subject.get_image(path)
        self.assertNPEqual(image, output)

    def test_context_opens_and_closes_file(self):
        original_cls = h5py.File
        opened = False

        def close_patch(original):
            def patch(*args, **kwargs):
                nonlocal opened
                opened = False
                original(*args, **kwargs)
            return patch

        def file_patch(*args, **kwargs):
            nonlocal opened
            opened = True
            file = original_cls(*args, **kwargs)
            file.close = close_patch(file.close)
            return file

        subject = self.make_instance()
        with unittest.mock.patch('arvet.database.image_manager.h5py.File', file_patch):
            self.assertFalse(opened)
            with subject:
                self.assertTrue(opened)
            self.assertFalse(opened)


    def test_handles_nested_contexts(self):
        subject = self.make_instance()
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path = subject.store_image(image)

        with subject:
            with subject:
                output = subject.get_image(path)
                self.assertNPEqual(image, output)
            output = subject.get_image(path)
            self.assertNPEqual(image, output)
