import unittest
import abc
import numpy as np
import xxhash
import core.image
import util.transform as tf
import metadata.image_metadata as imeta
import image_collections.image_augmentations.simple_augmentations as simp


# A simple mixin for some common tests to all simple augmentations
class TestSimpleAugmentation(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def do_augment(self, image):
        pass

    @abc.abstractmethod
    def get_test_projected_points(self):
        return []

    def test_sets_base_image_and_transformation_matrix(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertEqual(image, result.metadata.base_image)
        matrix = result.metadata.affine_transformation_matrix
        for point, point_prime in self.get_test_projected_points():
            point = (point[0], point[1], 1)
            calc_point = np.dot(matrix, point)
            self.assertNPEqual(point_prime[0:2], calc_point[0:2])

    def test_stacks_base_image_and_transformation_matrix(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        base_image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        # Create a transformation that is a rotation by 30 degrees around the centre
        a = np.cos(np.pi / 3)
        b = np.sin(np.pi / 3)
        transformation_matrix = np.array([[a, b, (1 - a) * 50 - b * 50],
                                          [-b, a, b * 50 + (1 - a) * 50],
                                          [0, 0, 1]])
        inv_transformation_matrix = np.linalg.inv(transformation_matrix)
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest(),
                base_image=base_image,
                transformation_matrix=transformation_matrix
            )
        )
        result = self.do_augment(image)
        self.assertEqual(base_image, result.metadata.base_image)
        matrix = result.metadata.affine_transformation_matrix
        for point, point_prime in self.get_test_projected_points():
            point = (point[0], point[1], 1)
            base_point = np.dot(inv_transformation_matrix, point)
            calc_point = np.dot(matrix, base_point)
            self.assertNPClose(point_prime[0:2], calc_point[0:2])

    def test_preserves_other_metadata(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                hash_=xxhash.xxh64(data).digest(),
                source_type=imeta.ImageSourceType.SYNTHETIC,
                environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                light_level=imeta.LightingLevel.WELL_LIT,
                time_of_day=imeta.TimeOfDay.DAY,

                width=data.shape[1],
                height=data.shape[0],
                camera_pose=tf.Transform((1, 3, 4), (0.2, 0.8, 0.2, -0.7)),
                right_camera_pose=tf.Transform((-10, -20, -30), (0.9, -0.7, 0.5, -0.3)),
                fov=90,
                focal_length=5,
                aperture=22,

                simulation_world='TestSimulationWorld',
                lighting_model=imeta.LightingModel.LIT,
                texture_mipmap_bias=1,
                normal_maps_enabled=True,
                roughness_enabled=True,
                geometry_decimation=0.8,

                procedural_generation_seed=16234,

                labelled_objects=[
                    imeta.LabelledObject(
                        class_names=('cup',),
                        bounding_box=(142, 280, 54, 78),
                        label_color=(2, 227, 34),
                        relative_pose=tf.Transform(location=(-246, 468, 4), rotation=(0.2, 0.3, 0.4)),
                        object_id='LabelledObject-68478'
                    ),
                    imeta.LabelledObject(
                        class_names=('car',),
                        bounding_box=(542, 83, 63, 123),
                        label_color=(26, 12, 212),
                        relative_pose=tf.Transform(location=(61, -717, 161), rotation=(0.7, 0.6, 0.3)),
                        object_id='LabelledObject-8246'
                    ),
                    imeta.LabelledObject(
                        class_names=('cow',),
                        bounding_box=(349, 672, 124, 208),
                        label_color=(162, 134, 163),
                        relative_pose=tf.Transform(location=(286, -465, -165), rotation=(0.9, 0.1, 0.5)),
                        object_id='LabelledObject-56485'
                    )
                ],
                average_scene_depth=90.12
            )
        )
        result = self.do_augment(image)
        self.assertEqual(result.metadata.source_type, image.metadata.source_type)
        self.assertEqual(result.metadata.environment_type, image.metadata.environment_type)
        self.assertEqual(result.metadata.light_level, image.metadata.light_level)
        self.assertEqual(result.metadata.time_of_day, image.metadata.time_of_day)
        self.assertEqual(result.metadata.height, image.metadata.height)
        self.assertEqual(result.metadata.width, image.metadata.width)
        self.assertEqual(result.metadata.camera_pose, image.metadata.camera_pose)
        self.assertEqual(result.metadata.right_camera_pose, image.metadata.right_camera_pose)
        self.assertEqual(result.metadata.fov, image.metadata.fov)
        self.assertEqual(result.metadata.focal_length, image.metadata.focal_length)
        self.assertEqual(result.metadata.aperture, image.metadata.aperture)
        self.assertEqual(result.metadata.simulation_world, image.metadata.simulation_world)
        self.assertEqual(result.metadata.lighting_model, image.metadata.lighting_model)
        self.assertEqual(result.metadata.texture_mipmap_bias, image.metadata.texture_mipmap_bias)
        self.assertEqual(result.metadata.normal_maps_enabled, image.metadata.normal_maps_enabled)
        self.assertEqual(result.metadata.roughness_enabled, image.metadata.roughness_enabled)
        self.assertEqual(result.metadata.geometry_decimation, image.metadata.geometry_decimation)
        self.assertEqual(result.metadata.procedural_generation_seed, image.metadata.procedural_generation_seed)
        self.assertEqual(result.metadata.average_scene_depth, image.metadata.average_scene_depth)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestHorizontalFlip(TestSimpleAugmentation, unittest.TestCase):

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

    def test_flips_horizontally(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertIsInstance(result, core.image.Image)
        self.assertNPEqual(np.array([list(range(i + 99, i - 1, -1)) for i in range(100)]), result.data)

    def test_flips_metadata(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        labels_data = np.array([list(range(i + 1, i + 101)) for i in range(100)])
        depth_data = np.array([list(range(i + 2, i + 102)) for i in range(100)])
        world_normals_data = np.array([list(range(i + 3, i + 103)) for i in range(100)])
        image = core.image.Image(
            data=data,
            labels_data=labels_data,
            depth_data=depth_data,
            world_normals_data=world_normals_data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertNPEqual(np.array([list(range(i + 100, i, -1)) for i in range(100)]), result.labels_data)
        self.assertNPEqual(np.array([list(range(i + 101, i + 1, -1)) for i in range(100)]), result.depth_data)
        self.assertNPEqual(np.array([list(range(i + 102, i + 2, -1)) for i in range(100)]), result.world_normals_data)

    def test_flips_bounding_boxes(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest(),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 10)
                ), imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(10, 60, 10, 10)
                )]
            )
        )
        result = self.do_augment(image)
        self.assertEqual((10, 20, 10, 10), result.metadata.labelled_objects[0].bounding_box)
        self.assertEqual((80, 60, 10, 10), result.metadata.labelled_objects[1].bounding_box)

    def test_inverts_itself(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(self.do_augment(image))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))


class TestVerticalFlip(TestSimpleAugmentation, unittest.TestCase):

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

    def test_flips_horizontally(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertIsInstance(result, core.image.Image)
        self.assertNPEqual(np.array([list(range(i, i + 100)) for i in range(99, -1, -1)]), result.data)

    def test_flips_metadata(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        labels_data = np.array([list(range(i + 1, i + 101)) for i in range(100)])
        depth_data = np.array([list(range(i + 2, i + 102)) for i in range(100)])
        world_normals_data = np.array([list(range(i + 3, i + 103)) for i in range(100)])
        image = core.image.Image(
            data=data,
            labels_data=labels_data,
            depth_data=depth_data,
            world_normals_data=world_normals_data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertNPEqual(np.array([list(range(i + 1, i + 101)) for i in range(99, -1, -1)]), result.labels_data)
        self.assertNPEqual(np.array([list(range(i + 2, i + 102)) for i in range(99, -1, -1)]), result.depth_data)
        self.assertNPEqual(np.array([list(range(i + 3, i + 103)) for i in range(99, -1, -1)]),
                           result.world_normals_data)

    def test_flips_bounding_boxes(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest(),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 10)
                ), imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(10, 60, 10, 10)
                )]
            )
        )
        result = self.do_augment(image)
        self.assertEqual((80, 70, 10, 10), result.metadata.labelled_objects[0].bounding_box)
        self.assertEqual((10, 30, 10, 10), result.metadata.labelled_objects[1].bounding_box)

    def test_inverts_itself(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(self.do_augment(image))
        self.assertNPEqual(result.data, data)
        self.assertNPEqual(result.metadata.affine_transformation_matrix, np.identity(3))


class TestRotate90(TestSimpleAugmentation, unittest.TestCase):

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

    def test_rotates_90_anticlockwise(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertIsInstance(result, core.image.Image)
        self.assertNPEqual(np.array([list(range(i, i + 100)) for i in range(99, -1, -1)]), result.data)

    def test_rotates_metadata(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        labels_data = np.array([list(range(i + 1, i + 101)) for i in range(100)])
        depth_data = np.array([list(range(i + 2, i + 102)) for i in range(100)])
        world_normals_data = np.array([list(range(i + 3, i + 103)) for i in range(100)])
        image = core.image.Image(
            data=data,
            labels_data=labels_data,
            depth_data=depth_data,
            world_normals_data=world_normals_data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertNPEqual(np.array([list(range(i + 1, i + 101)) for i in range(99, -1, -1)]), result.labels_data)
        self.assertNPEqual(np.array([list(range(i + 2, i + 102)) for i in range(99, -1, -1)]), result.depth_data)
        self.assertNPEqual(np.array([list(range(i + 3, i + 103)) for i in range(99, -1, -1)]),
                           result.world_normals_data)

    def test_rotates_bounding_boxes(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest(),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                ), imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(10, 60, 10, 20)
                )]
            )
        )
        result = self.do_augment(image)
        self.assertEqual((20, 10, 20, 10), result.metadata.labelled_objects[0].bounding_box)
        self.assertEqual((60, 80, 20, 10), result.metadata.labelled_objects[1].bounding_box)


class TestRotate180(TestSimpleAugmentation, unittest.TestCase):

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

    def test_rotates_180_anticlockwise(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertIsInstance(result, core.image.Image)
        self.assertNPEqual(np.array([list(range(i + 99, i - 1, -1)) for i in range(99, -1, -1)]), result.data)

    def test_rotates_metadata(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        labels_data = np.array([list(range(i + 1, i + 101)) for i in range(100)])
        depth_data = np.array([list(range(i + 2, i + 102)) for i in range(100)])
        world_normals_data = np.array([list(range(i + 3, i + 103)) for i in range(100)])
        image = core.image.Image(
            data=data,
            labels_data=labels_data,
            depth_data=depth_data,
            world_normals_data=world_normals_data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertNPEqual(np.array([list(range(i + 100, i, -1)) for i in range(99, -1, -1)]), result.labels_data)
        self.assertNPEqual(np.array([list(range(i + 101, i + 1, -1)) for i in range(99, -1, -1)]), result.depth_data)
        self.assertNPEqual(np.array([list(range(i + 102, i + 2, -1)) for i in range(99, -1, -1)]),
                           result.world_normals_data)

    def test_rotates_bounding_boxes(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest(),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                ), imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(10, 60, 10, 20)
                )]
            )
        )
        result = self.do_augment(image)
        self.assertEqual((10, 60, 10, 20), result.metadata.labelled_objects[0].bounding_box)
        self.assertEqual((80, 20, 10, 20), result.metadata.labelled_objects[1].bounding_box)


class TestRotate270(TestSimpleAugmentation, unittest.TestCase):

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

    def test_rotates_270_anticlockwise(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertIsInstance(result, core.image.Image)
        self.assertNPEqual(np.array([list(range(i + 99, i - 1, -1)) for i in range(100)]), result.data)

    def test_rotates_metadata(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        labels_data = np.array([list(range(i + 1, i + 101)) for i in range(100)])
        depth_data = np.array([list(range(i + 2, i + 102)) for i in range(100)])
        world_normals_data = np.array([list(range(i + 3, i + 103)) for i in range(100)])
        image = core.image.Image(
            data=data,
            labels_data=labels_data,
            depth_data=depth_data,
            world_normals_data=world_normals_data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest()
            )
        )
        result = self.do_augment(image)
        self.assertNPEqual(np.array([list(range(i + 100, i, -1)) for i in range(100)]), result.labels_data)
        self.assertNPEqual(np.array([list(range(i + 101, i + 1, -1)) for i in range(100)]), result.depth_data)
        self.assertNPEqual(np.array([list(range(i + 102, i + 2, -1)) for i in range(100)]),
                           result.world_normals_data)

    def test_rotates_bounding_boxes(self):
        data = np.array([list(range(i, i + 100)) for i in range(100)])
        image = core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                source_type=imeta.ImageSourceType.SYNTHETIC,
                width=data.shape[1],
                height=data.shape[0],
                hash_=xxhash.xxh64(data).digest(),
                labelled_objects=[imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(80, 20, 10, 20)
                ), imeta.LabelledObject(
                    class_names={'cup'},
                    bounding_box=(10, 60, 10, 20)
                )]
            )
        )
        result = self.do_augment(image)
        self.assertEqual((60, 80, 20, 10), result.metadata.labelled_objects[0].bounding_box)
        self.assertEqual((20, 10, 20, 10), result.metadata.labelled_objects[1].bounding_box)
