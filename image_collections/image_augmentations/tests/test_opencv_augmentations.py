#Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import core.image
import metadata.image_metadata as imeta
import image_collections.tests.test_augmented_collection as test_augmented
import image_collections.image_augmentations.opencv_augmentations as cv_aug


class TestRotateSimple(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = cv_aug.Rotate(np.pi / 2, 0.5, 0.5)
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

    def is_precise(self):
        return False

    def test_repeated_applications_returns_to_original(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=b'\xa5\xc9\x08\xaf$\x0b\x116',
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                )]
            )
        )
        # Apply 4 times to rotate back to the start
        result = self.do_augment(self.do_augment(self.do_augment(self.do_augment(image))))
        self.assertNPEqual(result.data, data)
        self.assertNPClose(result.metadata.affine_transformation_matrix, np.identity(3))
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)

        for _ in range(10):
            result = self.do_augment(self.do_augment(self.do_augment(self.do_augment(result))))
            self.assertNPEqual(result.data, data)
            self.assertNPClose(result.metadata.affine_transformation_matrix, np.identity(3))
            self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[0].bounding_box)


class TestTranslate(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        subject = cv_aug.Translate(0.25, 0.25)
        return subject.augment(image)

    def get_test_projected_points(self):
        return [
            ((0, 0), (25, 25)),
            ((99, 0), (124, 25)),
            ((0, 99), (25, 124)),
            ((99, 99), (124, 124))
        ]

    def create_base_and_augmented_data(self, seed=0):
        data = np.array([list(range(i + seed, i + seed + 100)) for i in range(100)])
        translated_data = np.zeros(data.shape)
        translated_data[25:100, 25:100] = data[0:75, 0:75]
        return data, translated_data

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 20), (105, 45, 10, 20)),
            ((10, 60, 10, 20), (35, 85, 10, 20))
        ]


class TestWarpAffine(test_augmented.ImageAugmenterContract, unittest.TestCase):

    def do_augment(self, image):
        projected_points = self.get_test_projected_points()
        resolution = np.array([image.data.shape[1], image.data.shape[0]])
        subject = cv_aug.WarpAffine(projected_points[0][0] / resolution, projected_points[0][1] / resolution,
                                    projected_points[1][0] / resolution, projected_points[1][1] / resolution,
                                    projected_points[2][0] / resolution, projected_points[2][1] / resolution)
        return subject.augment(image)

    def get_test_projected_points(self):
        # This is doing a combination rotate and translate
        return [
            ((0, 0), (25, 124)),
            ((99, 0), (25, 25)),
            ((0, 99), (124, 124)),
            ((99, 99), (124, 25))
        ]

    def create_base_and_augmented_data(self, seed=0):
        data = np.array([list(range(i + seed, i + seed + 100)) for i in range(100)])
        translated_data = np.zeros(data.shape)
        translated_data[25:100, 25:100] = np.rot90(data, k=1)[0:75, 0:75]
        return data, translated_data

    def get_projected_bounding_boxes(self):
        return [
            ((80, 20, 10, 20), (45, 34, 20, 10)),
            ((10, 60, 10, 20), (85, 104, 20, 10))
        ]
