# Copyright (c) 2017, John Skinner
import unittest

import arvet.core.image
import arvet.image_collections.image_augmentations.simple_augmentations as simp
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import numpy as np

import arvet.image_collections.tests.test_augmented_collection as test_augmented


class TestHorizontalFlip(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = simp.HorizontalFlip()
        return subject.augment(image)

    def get_test_projected_points(self):
        return [
            ((0, 0), (99, 0)),
            ((99, 0), (0, 0)),
            ((0, 99), (99, 99)),
            ((99, 99), (0, 99))
        ]

    def create_base_and_augmented_data(self, seed=0):
        return (np.array([list(range(i + seed, i + seed + 100)) for i in range(100)]),
                np.array([list(range(i + seed + 99, i + seed - 1, -1)) for i in range(100)]))

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 10), (9, 20, 10, 10)),
            ((10, 60, 10, 10), (79, 60, 10, 10))
        ]

    def test_inverts_itself(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = arvet.core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                hash_=b'\xa5\xc9\x08\xaf$\x0b\x116',
                intrinsics=cam_intr.CameraIntrinsics(100, 100, 55.2, 53.2, 50, 50),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                )]
            )
        )
        result = self.do_augment(self.do_augment(image))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)


class TestVerticalFlip(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = simp.VerticalFlip()
        return subject.augment(image)

    def get_test_projected_points(self):
        return [
            ((0, 0), (0, 99)),
            ((99, 0), (99, 99)),
            ((0, 99), (0, 0)),
            ((99, 99), (99, 0))
        ]

    def create_base_and_augmented_data(self, seed=0):
        return (np.array([list(range(i + seed, i + seed + 100)) for i in range(100)]),
                np.array([list(range(i + seed, i + seed + 100)) for i in range(99, -1, -1)]))

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 10), (80, 69, 10, 10)),
            ((10, 60, 10, 10), (10, 29, 10, 10))
        ]

    def test_inverts_itself(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = arvet.core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                hash_=b'\xa5\xc9\x08\xaf$\x0b\x116',
                intrinsics=cam_intr.CameraIntrinsics(100, 100, 55.2, 53.2, 50, 50),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                )]
            )
        )
        result = self.do_augment(self.do_augment(image))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)


class TestRotate90(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = simp.Rotate90()
        return subject.augment(image)

    def get_test_projected_points(self):
        return [
            ((0, 0), (0, 99)),
            ((99, 0), (0, 0)),
            ((0, 99), (99, 99)),
            ((99, 99), (99, 0))
        ]

    def create_base_and_augmented_data(self, seed=0):
        return (np.array([list(range(i + seed, i + seed + 100)) for i in range(100)]),
                np.array([list(range(i + seed, i + seed + 100)) for i in range(99, -1, -1)]))

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 20), (20, 9, 20, 10)),
            ((10, 60, 10, 20), (60, 79, 20, 10))
        ]

    def test_repeated_applications(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = arvet.core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                hash_=b'\xa5\xc9\x08\xaf$\x0b\x116',
                intrinsics=cam_intr.CameraIntrinsics(100, 100, 55.2, 53.2, 50, 50),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                )]
            )
        )
        # Apply 4 times to rotate back to the start
        result = self.do_augment(self.do_augment(self.do_augment(self.do_augment(image))))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)


class TestRotate180(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = simp.Rotate180()
        return subject.augment(image)

    def get_test_projected_points(self):
        return [
            ((0, 0), (99, 99)),
            ((99, 0), (0, 99)),
            ((0, 99), (99, 0)),
            ((99, 99), (0, 0))
        ]

    def create_base_and_augmented_data(self, seed=0):
        return (np.array([list(range(i + seed, i + seed + 100)) for i in range(100)]),
                np.array([list(range(i + seed + 99, i + seed - 1, -1)) for i in range(99, -1, -1)]))

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 20), (9, 59, 10, 20)),
            ((10, 60, 10, 20), (79, 19, 10, 20))
        ]

    def test_repeated_applications(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = arvet.core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                hash_=b'\xa5\xc9\x08\xaf$\x0b\x116',
                intrinsics=cam_intr.CameraIntrinsics(100, 100, 55.2, 53.2, 50, 50),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                )]
            )
        )
        # Apply 4 times to rotate back to the start
        result = self.do_augment(self.do_augment(self.do_augment(self.do_augment(image))))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)


class TestRotate270(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = simp.Rotate270()
        return subject.augment(image)

    def get_test_projected_points(self):
        return [
            ((0, 0), (99, 0)),
            ((0, 99), (0, 0)),
            ((99, 0), (99, 99)),
            ((99, 99), (0, 99))
        ]

    def create_base_and_augmented_data(self, seed=0):
        return (np.array([list(range(i + seed, i + seed + 100)) for i in range(100)]),
                np.array([list(range(i + seed + 99, i + seed - 1, -1)) for i in range(100)]))

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 20), (59, 80, 20, 10)),
            ((10, 60, 10, 20), (19, 10, 20, 10))
        ]

    def test_repeated_applications(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = arvet.core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                hash_=b'\xa5\xc9\x08\xaf$\x0b\x116',
                intrinsics=cam_intr.CameraIntrinsics(100, 100, 55.2, 53.2, 50, 50),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                )]
            )
        )
        # Apply 4 times to rotate back to the start
        result = self.do_augment(self.do_augment(self.do_augment(self.do_augment(image))))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)
