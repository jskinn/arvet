# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import abc
import numpy as np
import pymongo.collection
import bson.objectid
import arvet.database.tests.test_entity
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import arvet.core.image
import arvet.core.sequence_type
import arvet.core.image_collection
import arvet.image_collections.augmented_collection as aug_coll
import arvet.image_collections.image_augmentations.simple_augmentations as simple_augments
import arvet.util.dict_utils as du
import arvet.util.transform as tf


class TestAugmentedCollection(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return aug_coll.AugmentedImageCollection

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'inner': make_image_collection(),
            'augmenters': [simple_augments.HorizontalFlip(), simple_augments.Rotate270(), None]
        })
        return aug_coll.AugmentedImageCollection(*args, **kwargs)

    def assert_models_equal(self, collection1, collection2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param collection1:
        :param collection2:
        :return:
        """
        if (not isinstance(collection1, aug_coll.AugmentedImageCollection) or
                not isinstance(collection2, aug_coll.AugmentedImageCollection)):
            self.fail('object was not an AugmentedImageCollection')
        self.assertEqual(collection1.identifier, collection2.identifier)
        self.assertEqual(collection1._inner.identifier, collection2._inner.identifier)
        self.assertEqual(len(collection1._augmenters), len(collection2._augmenters))
        for idx in range(len(collection1._augmenters)):
            # Compare augmenters by serialized representation, we don't have a good approach here
            if collection1._augmenters[idx] is None:
                self.assertIsNone(collection2._augmenters[idx])
            else:
                self.assertIsNotNone(collection2._augmenters[idx])
                self.assertEqual(collection1._augmenters[idx].serialize(), collection2._augmenters[idx].serialize())

    def create_mock_db_client(self):
        db_client = super().create_mock_db_client()
        db_client.image_source_collection = mock.create_autospec(pymongo.collection.Collection)
        db_client.image_source_collection.find_one.side_effect = lambda q: {
            '_type': 'arvet.core.image_collection.ImageCollection',
            '_id': q['_id']
        }
        db_client.deserialize_entity.side_effect = mock_deserialize_entity
        return db_client

    def test_begin_calls_begin_on_inner(self):
        inner = mock.create_autospec(arvet.core.image_collection.ImageCollection)
        inner.get_next_image.return_value = (make_image(), 10)
        subject = aug_coll.AugmentedImageCollection(inner, [None, simple_augments.HorizontalFlip()])
        subject.begin()
        self.assertTrue(inner.begin.called)

    def test_returns_augmented_images_in_order(self):
        img1 = make_image(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        img2 = make_image(data=np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]]))
        collection = make_image_collection(images={1: img1, 2: img2})
        subject = aug_coll.AugmentedImageCollection(collection, [
            simple_augments.HorizontalFlip(), simple_augments.VerticalFlip()])

        subject.begin()
        img, _ = subject.get_next_image()
        self.assertNPEqual([[3, 2, 1], [6, 5, 4], [9, 8, 7]], img.data)
        img, _ = subject.get_next_image()
        self.assertNPEqual([[7, 8, 9], [4, 5, 6], [1, 2, 3]], img.data)
        img, _ = subject.get_next_image()
        self.assertNPEqual([[13, 12, 11], [16, 15, 14], [19, 18, 17]], img.data)
        img, _ = subject.get_next_image()
        self.assertNPEqual([[17, 18, 19], [14, 15, 16], [11, 12, 13]], img.data)
        self.assertTrue(subject.is_complete())

    def test_single_null_augment_is_same_as_inner_collection(self):
        images = {idx + np.random.uniform(-0.2, 0.2):
                  make_image(data=np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)) for idx in range(10)}
        collection = make_image_collection(images=images)
        subject = aug_coll.AugmentedImageCollection(collection, [None])

        self.assertEqual(10, len(subject))
        loop_count = 0
        prev_stamp = -1
        subject.begin()
        while not subject.is_complete():
            img, stamp = subject.get_next_image()
            self.assertIn(stamp, images)
            self.assertNPEqual(images[stamp].data, img.data)
            self.assertGreater(stamp, prev_stamp)
            prev_stamp = stamp
            loop_count += 1
        self.assertEqual(loop_count, 10)

    def test_provides_index_as_stamp_if_multiple_augmenters(self):
        images = {idx + np.random.uniform(-0.2, 0.2):
                  make_image(data=np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)) for idx in range(10)}
        collection = make_image_collection(images=images)
        subject = aug_coll.AugmentedImageCollection(collection, [
            simple_augments.HorizontalFlip(), simple_augments.VerticalFlip()])

        self.assertEqual(20, len(subject))
        subject.begin()
        for idx in range(20):
            _, stamp = subject.get_next_image()
            self.assertEqual(idx, stamp)
        self.assertTrue(subject.is_complete())

    def test_multiple_augmenters_makes_non_sequential(self):
        images = {idx + np.random.uniform(-0.2, 0.2):
                  make_image(data=np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)) for idx in range(10)}
        collection = make_image_collection(images=images, type_=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        subject = aug_coll.AugmentedImageCollection(collection, [simple_augments.Rotate270()])
        self.assertEqual(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL, subject.sequence_type)
        subject = aug_coll.AugmentedImageCollection(collection, [simple_augments.Rotate270(),
                                                                 simple_augments.Rotate90()])
        self.assertEqual(arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, subject.sequence_type)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class ImageAugmenterContract(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def do_augment(self, image):
        """
        Do the augmentation
        :param image: An image object
        :return: A modified image object
        """
        pass

    @abc.abstractmethod
    def get_test_projected_points(self):
        """
        Return a list of pairs of points, base point to augmented point.
        Structure is [((1,2), (3,4)), ((5,6), (7,8))], indicating that
        (1,2) -> (3,4) and (5,6) -> (7,8)
        :return:
        """
        return []

    @abc.abstractmethod
    def create_base_and_augmented_data(self, seed=0):
        """
        Create a pair of
        :param seed: A seed to make the data different, doesn't have to be random
        :return: Two numpy arrays, containing base image and augmented image
        """
        return np.array([]), np.array([])

    @abc.abstractmethod
    def get_projected_bounding_boxes(self):
        """
        Get a list of pairs of bounding boxes, one for base image and one for augmented image.
        Bounding boxes should be (x, y, width, height)
        :return:
        """
        return []

    def is_precise(self):
        """
        Should we use precise comparisons for projected points.
        Some projections are inherently a little imprecise (such as rotating by 45 degrees),
        so some floating-point error when projecting is acceptable.
        :return:
        """
        return False

    def test_sets_base_image_and_transformation_matrix(self):
        image = make_image()
        result = self.do_augment(image)
        self.assertEqual(image, result.metadata.base_image)
        matrix = result.metadata.affine_transformation_matrix
        for point, point_prime in self.get_test_projected_points():
            point = (point[0], point[1], 1)
            calc_point = np.dot(matrix, point)
            if self.is_precise():
                self.assertNPEqual(point_prime[0:2], calc_point[0:2])
            else:
                self.assertNPClose(point_prime[0:2], calc_point[0:2])

    def test_stacks_base_image_and_transformation_matrix(self):
        base_image = make_image()
        # Create a transformation that is a rotation by 30 degrees around the centre
        a = np.cos(np.pi / 3)
        b = np.sin(np.pi / 3)
        transformation_matrix = np.array([[a, b, (1 - a) * 50 - b * 50],
                                          [-b, a, b * 50 + (1 - a) * 50],
                                          [0, 0, 1]])
        inv_transformation_matrix = np.linalg.inv(transformation_matrix)
        image = make_image(metadata={
            'base_image': base_image,
            'transformation_matrix': transformation_matrix
        })
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
        image = arvet.core.image.Image(
            data=data,
            metadata=imeta.ImageMetadata(
                hash_=b'\x04\xe2\x1f\x3d$\x7c\x116',
                source_type=imeta.ImageSourceType.SYNTHETIC,
                environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
                light_level=imeta.LightingLevel.WELL_LIT,
                time_of_day=imeta.TimeOfDay.DAY,

                camera_pose=tf.Transform((1, 3, 4), (0.2, 0.8, 0.2, -0.7)),
                right_camera_pose=tf.Transform((-10, -20, -30), (0.9, -0.7, 0.5, -0.3)),
                intrinsics=cam_intr.CameraIntrinsics(data.shape[1], data.shape[0], 147.2, 123.3, 420, 215),
                right_intrinsics=cam_intr.CameraIntrinsics(data.shape[1], data.shape[0], 168.2, 123.3, 420, 251),
                lens_focal_distance=5,
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
        self.assertEqual(result.metadata.lens_focal_distance, image.metadata.lens_focal_distance)
        self.assertEqual(result.metadata.aperture, image.metadata.aperture)
        self.assertEqual(result.metadata.simulation_world, image.metadata.simulation_world)
        self.assertEqual(result.metadata.lighting_model, image.metadata.lighting_model)
        self.assertEqual(result.metadata.texture_mipmap_bias, image.metadata.texture_mipmap_bias)
        self.assertEqual(result.metadata.normal_maps_enabled, image.metadata.normal_maps_enabled)
        self.assertEqual(result.metadata.roughness_enabled, image.metadata.roughness_enabled)
        self.assertEqual(result.metadata.geometry_decimation, image.metadata.geometry_decimation)
        self.assertEqual(result.metadata.procedural_generation_seed, image.metadata.procedural_generation_seed)
        self.assertEqual(result.metadata.average_scene_depth, image.metadata.average_scene_depth)

    def test_modifies_image(self):
        data, augmented_data = self.create_base_and_augmented_data()
        image = make_image(data=data)
        result = self.do_augment(image)
        self.assertIsInstance(result, arvet.core.image.Image)
        self.assertNPEqual(augmented_data, result.data)

    # TODO: Augmenters should support stereo images as well
    #def test_modifies_stereo_image(self):

    def test_modifies_metadata_images(self):
        data, augmented_data = self.create_base_and_augmented_data(0)
        labels_data, augmented_labels_data = self.create_base_and_augmented_data(1)
        depth_data, augmented_depth_data = self.create_base_and_augmented_data(2)
        normals_data, augmented_normals_data = self.create_base_and_augmented_data(3)
        image = make_image(
            data=data,
            labels_data=labels_data,
            depth_data=depth_data,
            world_normals_data=normals_data
        )
        result = self.do_augment(image)
        self.assertNPEqual(augmented_labels_data, result.labels_data)
        self.assertNPEqual(augmented_depth_data, result.depth_data)
        self.assertNPEqual(augmented_normals_data, result.world_normals_data)

    def test_modifies_bounding_boxes(self):
        labelled_objects = []
        modified_bboxes = []
        for bbox, bbox_prime in self.get_projected_bounding_boxes():
            labelled_objects.append(imeta.LabelledObject(
                class_names={'cup'},
                bounding_box=bbox
            ))
            modified_bboxes.append(bbox_prime)

        image = make_image(metadata={'labelled_objects': labelled_objects})
        result = self.do_augment(image)
        self.assertEqual(len(modified_bboxes), len(result.metadata.labelled_objects))
        for idx in range(len(modified_bboxes)):
            self.assertEqual(modified_bboxes[idx], result.metadata.labelled_objects[idx].bounding_box)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def make_image(**kwargs):
    if 'data' in kwargs:
        data = kwargs['data']
    else:
        data = np.array([list(range(i, i + 100)) for i in range(100)])
    metadata_kwargs = {
        'source_type': imeta.ImageSourceType.SYNTHETIC,
        'hash_': b'\xa5\xc9\x08\xaf$\x0b\x116',
        'intrinsics': cam_intr.CameraIntrinsics(100, 100, 55.2, 53.2, 50, 50)
    }
    if 'metadata' in kwargs and isinstance(kwargs['metadata'], dict):
        metadata_kwargs = du.defaults(kwargs['metadata'], metadata_kwargs)
        del kwargs['metadata']
    kwargs = du.defaults(kwargs, {
        'data': data,
        'metadata': imeta.ImageMetadata(**metadata_kwargs)
    })
    return arvet.core.image.Image(**kwargs)


def make_image_collection(**kwargs):
    if 'images' not in kwargs:
        kwargs['images'] = {1: make_image()}
    du.defaults(kwargs, {
        'type_': arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
        'id_': bson.ObjectId()
    })
    return arvet.core.image_collection.ImageCollection(**kwargs)


def mock_deserialize_entity(s_entity):
    if s_entity['_type'] == 'arvet.core.image_collection.ImageCollection':
        return make_image_collection(id_=s_entity['_id'])
    elif s_entity['_type'] == 'arvet.image_collections.image_augmentations.simple_augmentations.HorizontalFlip':
        return simple_augments.HorizontalFlip()
    elif s_entity['_type'] == 'arvet.image_collections.image_augmentations.simple_augmentations.VerticalFlip':
        return simple_augments.VerticalFlip()
    elif s_entity['_type'] == 'arvet.image_collections.image_augmentations.simple_augmentations.Rotate90':
        return simple_augments.Rotate90()
    elif s_entity['_type'] == 'arvet.image_collections.image_augmentations.simple_augmentations.Rotate180':
        return simple_augments.Rotate180()
    elif s_entity['_type'] == 'arvet.image_collections.image_augmentations.simple_augmentations.Rotate270':
        return simple_augments.Rotate270()
    return None
