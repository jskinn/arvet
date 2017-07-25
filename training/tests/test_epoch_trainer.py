import unittest
import numpy as np
import bson.objectid
import util.dict_utils as du
import util.transform as tf
import database.tests.test_entity
import metadata.image_metadata as imeta
import core.image_entity as ie
import core.image_collection as ic
import core.sequence_type
import core.tests.test_trained_system
import training.epoch_trainer


class TestEpochTrainer(core.tests.test_trained_system.TrainerContract,
                       database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.image_map = {}
        self.images_list = []
        for i in range(10):
            image = make_image()
            self.image_map[str(image.identifier)] = image
            self.images_list.append(image)

    def get_class(self):
        return training.epoch_trainer.EpochTrainer

    def make_instance(self, *args, **kwargs):
        self.make_image_collection_map()
        kwargs = du.defaults(kwargs, {
            'image_sources': list(self.image_collection_map.values()),
            'num_epochs': np.random.randint(0, 10),
            'use_source_length': bool(np.random.randint(0, 2)),
            'epoch_length': np.random.randint(0, 10),
            'horizontal_flips': bool(np.random.randint(0, 2)),
            'vertical_flips': bool(np.random.randint(0, 2)),
            'rot_90': bool(np.random.randint(0, 2)),
            'validation_fraction': np.random.uniform(0, 1)
        })
        return training.epoch_trainer.EpochTrainer(*args, **kwargs)

    def assert_models_equal(self, result1, result2):
        """
        Helper to assert that two epoch trainers are equal
        :param result1:
        :param result2:
        :return:
        """
        if (not isinstance(result1, training.epoch_trainer.EpochTrainer) or
                not isinstance(result2, training.epoch_trainer.EpochTrainer)):
            self.fail('object was not a EpochTrainer')
        self.assertEqual(result1.identifier, result2.identifier)
        self.assertEqual(result1.num_image_sources, result2.num_image_sources)
        self.assertEqual(result1._num_epochs, result2._num_epochs)
        self.assertEqual(result1._epoch_length, result2._epoch_length)
        self.assertEqual(result1._use_image_source_length, result2._use_image_source_length)
        self.assertEqual(result1._horizontal_flips, result2._horizontal_flips)
        self.assertEqual(result1._vertical_flips, result2._vertical_flips)
        self.assertEqual(result1._rot_90, result2._rot_90)
        self.assertEqual(result1._validation_fraction, result2._validation_fraction)
        for source_idx in range(result1.num_image_sources):
            image_collection1 = result1._image_sources[source_idx]
            image_collection2 = result2._image_sources[source_idx]
            self.assertEqual(image_collection1.identifier, image_collection2.identifier)
            self.assertEqual(image_collection1.sequence_type, image_collection2.sequence_type)
            self.assertEqual(image_collection1.is_depth_available, image_collection2.is_depth_available)
            self.assertEqual(image_collection1.is_per_pixel_labels_available,
                             image_collection2.is_per_pixel_labels_available)
            self.assertEqual(image_collection1.is_normals_available, image_collection2.is_normals_available)
            self.assertEqual(image_collection1.is_stereo_available, image_collection2.is_stereo_available)
            self.assertEqual(len(image_collection1), len(image_collection2))
            for img_idx in range(len(image_collection1)):
                # Don't compare image details, if the ids are the same, they're the same
                self.assertEqual(image_collection1[img_idx].identifier, image_collection2[img_idx].identifier)

    def create_mock_db_client(self):
        self.db_client = super().create_mock_db_client()
        self.make_image_collection_map()

        serialized_collections = [{'_id': coll.identifier} for coll in self.image_collection_map.values()]
        self.db_client.image_source_collection.find.return_value = serialized_collections
        self.db_client.deserialize_entity.side_effect = lambda s_coll: self.image_collection_map[str(s_coll['_id'])]
        return self.db_client

    def make_image_collection_map(self):
        if not hasattr(self, 'image_collection_map'):
            self.image_collection_map = {}
            for _ in range(3):
                image_collection = make_mock_image_collection(20)
                self.image_collection_map[str(image_collection.identifier)] = image_collection

    def test_calls_validate_each_epoch(self):
        collection = make_mock_image_collection(10)
        trainee = self.make_mock_trainee()
        subject = training.epoch_trainer.EpochTrainer(
            image_sources=[collection],
            num_epochs=13,
            horizontal_flips=False,
            vertical_flips=False,
            rot_90=False,
            validation_fraction=0.3
        )

        subject.train_vision_system(trainee)
        # 3 validation images from the 10 image collection
        self.assertEqual(13, trainee.start_validation.call_count)
        self.assertEqual(39, trainee.validate_with_image.call_count)
        self.assertEqual(13, trainee.finish_validation.call_count)
        # Check it gives the right argument to start validation
        for call_args in trainee.start_validation.call_args_list:
            self.assertEqual(3, call_args[1]['num_validation_images'])

    def test_calls_each_image_exactly_once_when_using_source_length(self):
        collection = make_mock_image_collection(10)
        trainee = self.make_mock_trainee()
        subject = training.epoch_trainer.EpochTrainer(
            image_sources=[collection],
            num_epochs=13,
            use_source_length=True,
            horizontal_flips=False,
            vertical_flips=False,
            rot_90=False,
            validation_fraction=0.3
        )
        subject.train_vision_system(trainee)
        # 7 training images per epoch, 13 epochs means 91 training calls
        self.assertEqual(1, trainee.start_training.call_count)
        self.assertEqual(91, trainee.start_training.call_args[1]['num_images'])
        self.assertEqual(91, trainee.train_with_image.call_count)
        self.assertEqual(1, trainee.finish_training.call_count)
        # Check that it passes the correct index to each call of train_with_image
        for idx, call_args in enumerate(trainee.train_with_image.call_args_list):
            self.assertEqual(idx, call_args[1]['index'])

    def test_can_use_a_fixed_number_of_images_per_epoch(self):
        collection = make_mock_image_collection(10)
        trainee = self.make_mock_trainee()
        subject = training.epoch_trainer.EpochTrainer(
            image_sources=[collection],
            num_epochs=5,
            use_source_length=False,
            epoch_length=11,
            horizontal_flips=False,
            vertical_flips=False,
            rot_90=False,
            validation_fraction=0.3
        )
        subject.train_vision_system(trainee)
        # 11 training images per epoch, 5 epochs means 55 training calls
        self.assertEqual(55, trainee.start_training.call_args[1]['num_images'])
        self.assertEqual(55, trainee.train_with_image.call_count)

    def test_can_stack_data_augmentations_for_massive_data(self):
        collection = make_mock_image_collection(3)
        trainee = self.make_mock_trainee()
        subject = training.epoch_trainer.EpochTrainer(
            image_sources=[collection],
            num_epochs=1,
            use_source_length=True,
            horizontal_flips=True,  # double data
            vertical_flips=True,    # double it again
            rot_90=True,            # 4 times data
            validation_fraction=0.3
        )
        subject.train_vision_system(trainee)
        # 3*2*2*4=48 training images per epoch, 1 epochs means 91 training calls
        self.assertEqual(48, trainee.start_training.call_args[1]['num_images'])
        self.assertEqual(48, trainee.train_with_image.call_count)

    def test_provides_images_in_a_different_order_each_epoch(self):
        collection = make_mock_image_collection(5)
        trainee = self.make_mock_trainee()
        subject = training.epoch_trainer.EpochTrainer(
            image_sources=[collection],
            num_epochs=5,
            use_source_length=True,
            horizontal_flips=False,
            vertical_flips=False,
            rot_90=False,
            validation_fraction=0
        )
        subject.train_vision_system(trainee)
        # 5 training images per epoch, 5 epochs means 91 training calls
        self.assertEqual(25, trainee.train_with_image.call_count)

        # Split the calls by epoch, producing lists of 'image' values
        orders = [[trainee.train_with_image.call_args_list[epoch * 5 + idx][0][0]
                   for idx in range(5)] for epoch in range(5)]

        # make sure the list of calls in each epoch are not the same
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertNotEqual(orders[i], orders[j])

    def test_create_serialized_makes_deserializeable_collection(self):
        db_client = self.create_mock_db_client()
        s_trainer1 = training.epoch_trainer.EpochTrainer.create_serialized(
            image_sources=[coll.identifier for coll in self.image_collection_map.values()],
            num_epochs=13,
            epoch_length=142,
            use_source_length=False,
            horizontal_flips=False,
            vertical_flips=True,
            rot_90=False,
            validation_fraction=0.3175172
        )
        trainer1 = training.epoch_trainer.EpochTrainer.deserialize(s_trainer1, db_client)

        self.assertEqual(s_trainer1['num_epochs'], trainer1._num_epochs)
        self.assertEqual(s_trainer1['epoch_length'], trainer1._epoch_length)
        self.assertEqual(s_trainer1['use_source_length'], trainer1._use_image_source_length)
        self.assertEqual(s_trainer1['horizontal_flips'], trainer1._horizontal_flips)
        self.assertEqual(s_trainer1['vertical_flips'], trainer1._vertical_flips)
        self.assertEqual(s_trainer1['rot_90'], trainer1._rot_90)
        self.assertEqual(s_trainer1['validation_fraction'], trainer1._validation_fraction)
        self.assertEqual(len(s_trainer1['image_sources']), trainer1.num_image_sources)
        for source_idx in range(trainer1.num_image_sources):
            self.assertEqual(s_trainer1['image_sources'][source_idx], trainer1._image_sources[source_idx].identifier)

        s_trainer_2 = trainer1.serialize()
        self.assert_serialized_equal(s_trainer1, s_trainer_2)


def make_image(**kwargs):
    """
    Make a mock image, randomly
    :param kwargs: Fixed kwargs to the constructor
    :return: a new image object
    """
    kwargs = du.defaults(kwargs, {
        'id_': bson.objectid.ObjectId(),
        'data': np.random.randint(0, 255, (32, 32, 3), dtype='uint8'),
        'metadata': imeta.ImageMetadata(
            hash_=b'\x1f`\xa8\x8aR\xed\x9f\x0b',
            source_type=imeta.ImageSourceType.SYNTHETIC,
            height=600, width=800,
            camera_pose=tf.Transform(location=np.random.uniform(-1000, 1000, 3),
                                     rotation=np.random.uniform(-1, 1, 4)),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT,
            time_of_day=imeta.TimeOfDay.DAY,
            fov=np.random.randint(10, 90),
            focal_length=np.random.uniform(10, 10000),
            aperture=np.random.uniform(1, 22),
            simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT,
            texture_mipmap_bias=np.random.randint(0, 8),
            normal_maps_enabled=bool(np.random.randint(0, 2)),
            roughness_enabled=bool(np.random.randint(0, 2)),
            geometry_decimation=np.random.uniform(0, 1),
            procedural_generation_seed=np.random.randint(10000),
            labelled_objects=(
                imeta.LabelledObject(class_names=('car',),
                                     bounding_box=tuple(np.random.randint(0, 100, 4)),
                                     label_color=tuple(np.random.randint(0, 255, 3)),
                                     relative_pose=tf.Transform(np.random.uniform(-1000, 1000, 3),
                                                                np.random.uniform(-1, 1, 4)),
                                     object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',),
                                     bounding_box=tuple(np.random.randint(0, 100, 4)),
                                     label_color=tuple(np.random.randint(0, 255, 4)),
                                     relative_pose=tf.Transform(np.random.uniform(-1000, 1000, 3),
                                                                np.random.uniform(-1, 1, 4)),
                                     object_id='cat-090')
            ),
            average_scene_depth=np.random.uniform(10000)),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 32, 'height': 32},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth_data': np.random.uniform(0, 1, (32, 32)),
        'labels_data': np.random.uniform(0, 1, (32, 32, 3)),
        'world_normals_data': np.random.uniform(0, 1, (32, 32, 3))
    })
    return ie.ImageEntity(**kwargs)


def make_mock_image_collection(num_images):
    return ic.ImageCollection(
        id_=bson.objectid.ObjectId(),
        type_=core.sequence_type.ImageSequenceType.NON_SEQUENTIAL,
        images=[make_image() for _ in range(num_images)]
    )
