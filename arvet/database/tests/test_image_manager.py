import os.path
import unittest.mock as mock
import numpy as np
import xxhash
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

    def test_get_image_for_missing_path_returns_none(self):
        subject = im_manager.DefaultImageManager(self._tempfile)
        self.assertIsNone(subject.get_image('notapath'))

    def test_store_and_get_image(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path = subject.store_image(image)
        output = subject.get_image(path)
        self.assertNPEqual(image, output)

    def test_storing_the_same_image_repeatedly_gives_the_same_path(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path1 = subject.store_image(image)
        path2 = subject.store_image(image)
        self.assertEqual(path1, path2)

    def test_is_valid_path_returns_true_only_for_valid_paths(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path = subject.store_image(image)

        self.assertTrue(subject.is_valid_path(path))
        self.assertFalse(subject.is_valid_path('notapath'))

    def test_remove_image_deletes_the_image(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image1 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        image2 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path1 = subject.store_image(image1)
        path2 = subject.store_image(image2)

        self.assertTrue(subject.is_valid_path(path1))
        self.assertTrue(subject.is_valid_path(path2))

        subject.remove_image(path1)

        self.assertFalse(subject.is_valid_path(path1))
        self.assertTrue(subject.is_valid_path(path2))
        self.assertIsNone(subject.get_image(path1))
        self.assertNPEqual(image2, subject.get_image(path2))

    def test_storing_raises_exception_if_writing_is_disabled(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=False)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        with self.assertRaises(RuntimeError):
            subject.store_image(image)

    def test_remove_image_raises_exception_if_writing_disabled(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image1 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        image2 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path1 = subject.store_image(image1)
        path2 = subject.store_image(image2)

        self.assertTrue(subject.is_valid_path(path1))
        self.assertTrue(subject.is_valid_path(path2))

        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=False)
        with self.assertRaises(RuntimeError):
            subject.remove_image(path1)

    @mock.patch('arvet.database.image_manager.xxhash', autospec=True)
    def test_can_use_groups_to_avoid_hash_collisions(self, mock_xxhash):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image1 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        image2 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))

        # Patch the hash function to force a hash collision
        digest = 'FFAA12'
        mock_hash64 = mock.MagicMock()
        mock_hash64.hexdigest.return_value = digest
        mock_xxhash.xxh64.return_value = mock_hash64

        group1 = 'mygroup1'
        group2 = '/mygroup2/'
        path1 = subject.store_image(image1, group1)
        path2 = subject.store_image(image2, group2)
        self.assertEqual(group1 + '/' + digest, path1)
        self.assertEqual('mygroup2/' + digest, path2)

        output = subject.get_image(path1)
        self.assertNPEqual(image1, output)

        output = subject.get_image(path2)
        self.assertNPEqual(image2, output)

    @mock.patch('arvet.database.image_manager.xxhash', autospec=True)
    def test_store_image_changes_seed_to_avoid_collisions(self, mock_xxhash):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image1 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        image2 = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))

        # Patch the hash function to force a hash collision for the same seed
        def mock_hash_func(_, seed=0):
            mock_hash64 = mock.MagicMock()
            mock_hash64.hexdigest.return_value = 'FFAA12' + str(seed)
            return mock_hash64

        mock_xxhash.xxh64.side_effect = mock_hash_func

        path1 = subject.store_image(image1)
        path2 = subject.store_image(image2)
        self.assertEqual('FFAA120', path1)
        self.assertEqual('FFAA121', path2)

        output = subject.get_image(path1)
        self.assertNPEqual(image1, output)

        output = subject.get_image(path2)
        self.assertNPEqual(image2, output)

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

        subject = im_manager.DefaultImageManager(self._tempfile)
        with mock.patch('arvet.database.image_manager.h5py.File', file_patch):
            self.assertFalse(opened)
            with subject:
                self.assertTrue(opened)
            self.assertFalse(opened)

    def test_context_closes_even_if_exception_raised(self):
        original_cls = h5py.File
        closed = False

        def close_patch(original):
            def patch(*args, **kwargs):
                nonlocal closed
                closed = True
                original(*args, **kwargs)
            return patch

        def file_patch(*args, **kwargs):
            file = original_cls(*args, **kwargs)
            file.close = close_patch(file.close)
            return file

        subject = im_manager.DefaultImageManager(self._tempfile)
        with mock.patch('arvet.database.image_manager.h5py.File', file_patch):
            self.assertFalse(closed)
            try:
                with subject:
                    raise ValueError()
            except ValueError:
                pass
            self.assertTrue(closed)

    def test_handles_nested_contexts(self):
        subject = im_manager.DefaultImageManager(self._tempfile, allow_write=True)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        path = subject.store_image(image)

        with subject:
            with subject:
                output = subject.get_image(path)
                self.assertNPEqual(image, output)
            output = subject.get_image(path)
            self.assertNPEqual(image, output)

    def test_find_path_for_image_no_groups(self):
        subject = im_manager.DefaultImageManager(self._tempfile)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        digest = xxhash.xxh64(image, seed=0).hexdigest()
        self.assertEqual(digest, subject.find_path_for_image(image))

    def test_find_path_for_image_with_manager_group(self):
        subject = im_manager.DefaultImageManager(self._tempfile, group_name='test')
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        digest = xxhash.xxh64(image, seed=0).hexdigest()
        self.assertEqual('test/' + digest, subject.find_path_for_image(image))

    def test_find_path_strips_preceding_slashes(self):
        subject = im_manager.DefaultImageManager(self._tempfile, group_name='/test/foo/')
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        digest = xxhash.xxh64(image, seed=0).hexdigest()
        self.assertEqual('test/foo/' + digest, subject.find_path_for_image(image))

    def test_find_path_for_image_with_image_group(self):
        subject = im_manager.DefaultImageManager(self._tempfile)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        digest = xxhash.xxh64(image, seed=0).hexdigest()
        self.assertEqual('mygroup/' + digest, subject.find_path_for_image(image, 'mygroup'))

    def test_find_path_strips_slashes_from_image_group(self):
        subject = im_manager.DefaultImageManager(self._tempfile)
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        digest = xxhash.xxh64(image, seed=0).hexdigest()
        self.assertEqual('mygroup/mysubgroup/' + digest, subject.find_path_for_image(image, '/mygroup/mysubgroup//'))

    def test_find_path_with_both_groups(self):
        subject = im_manager.DefaultImageManager(self._tempfile, group_name='/test/foo/')
        image = np.random.randint(0, 255, dtype=np.uint8, size=(64, 64, 3))
        digest = xxhash.xxh64(image, seed=0).hexdigest()
        self.assertEqual('test/foo/mygroup/mysubgroup/' + digest,
                         subject.find_path_for_image(image, '/mygroup/mysubgroup/'))
