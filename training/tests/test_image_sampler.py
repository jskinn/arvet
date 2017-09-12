#Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import training.image_sampler


class MockImageSource:

    def __init__(self, images):
        self.images = {1.24 * t: image for t, image in enumerate(images)}
        self.timestamps = list(self.images.keys())

    def __len__(self):
        return len(self.images)

    def get(self, index):
        return self.images[index]

    def begin(self):
        pass


class MockAugmenter:
    def augment(self, x):
        return x + 10


class TestImageSampler(unittest.TestCase):

    def test_constructor_splits_existing_image_sources(self):
        image_source_1 = MockImageSource(range(0, 10))
        image_source_2 = MockImageSource(range(100, 110))
        subject = training.image_sampler.ImageSampler((image_source_1, image_source_2), default_validation_fraction=0.3)
        self.assertEqual(14, subject.num_training)
        self.assertEqual(6, subject.num_validation)

    def test_augmenters_increase_data(self):
        subject = training.image_sampler.ImageSampler(
            image_sources=(MockImageSource(range(0, 10)),),
            augmenters=(lambda x: x + 10, lambda x: x + 20),
            default_validation_fraction=0.3
        )
        # Does not stack augmenters by default, linear increase
        self.assertEqual(21, subject.num_training)
        self.assertEqual(9, subject.num_validation)

    def test_augmenters_can_be_objects(self):
        subject = training.image_sampler.ImageSampler(
            image_sources=(MockImageSource(range(0, 10)),),
            augmenters=(MockAugmenter(),),
            default_validation_fraction=0
        )
        self.assertEqual(20, subject.num_training)
        self.assertIn(13, {subject.get(idx) for idx in range(subject.num_training)})

    def test_same_value_doesnt_appear_in_training_and_validation(self):
        subject = training.image_sampler.ImageSampler(
            image_sources=(MockImageSource(range(0, 10)),),
            augmenters=(lambda x: x + 10, lambda x: x + 20),
            default_validation_fraction=0.3
        )
        # We care about the base values, not the augmented ones, hence mod 10
        training_values = {subject.get(idx) % 10 for idx in range(subject.num_training)}
        for idx in range(subject.num_validation):
            self.assertNotIn(subject.get_validation(idx) % 10, training_values)

    def test_get_raises_index_error_when_out_of_range(self):
        subject = training.image_sampler.ImageSampler(
            image_sources=(MockImageSource(range(0, 10)),),
            default_validation_fraction=0.3, loop=False
        )
        with self.assertRaises(IndexError):
            subject.get(-1)
        with self.assertRaises(IndexError):
            subject.get(100)
        with self.assertRaises(IndexError):
            subject.get_validation(-1)
        with self.assertRaises(IndexError):
            subject.get_validation(100)

    def test_loop_allows_out_of_range_indexing(self):
        subject = training.image_sampler.ImageSampler(
            image_sources=(MockImageSource(range(0, 10)),),
            default_validation_fraction=0.3, loop=True
        )
        self.assertEqual(subject.get(6), subject.get(-1))
        self.assertEqual(subject.get(6), subject.get(13))
        self.assertEqual(subject.get_validation(2), subject.get_validation(-1))
        self.assertEqual(subject.get_validation(2), subject.get_validation(5))

    def test_amalgamates_image_sources(self):
        image_source_1 = MockImageSource(range(0, 10))
        image_source_2 = MockImageSource(range(10, 20))
        image_source_3 = MockImageSource(range(20, 30))
        subject = training.image_sampler.ImageSampler((image_source_1, image_source_2, image_source_3),
                                                      default_validation_fraction=0.3)
        training_values = {subject.get(idx) for idx in range(subject.num_training)}
        validation_values = {subject.get_validation(idx) for idx in range(subject.num_validation)}
        # Check both test and validation draw from all sources
        self.assertTrue(any(i in training_values for i in range(0, 10)))
        self.assertTrue(any(i in training_values for i in range(10, 20)))
        self.assertTrue(any(i in training_values for i in range(20, 30)))
        self.assertTrue(any(i in validation_values for i in range(0, 10)))
        self.assertTrue(any(i in validation_values for i in range(10, 20)))
        self.assertTrue(any(i in validation_values for i in range(20, 30)))

        # Test that all of all the sources appear between the two sets
        self.assertEqual({i for i in range(0, 30)}, training_values | validation_values)

    def test_shuffle_changes_order(self):
        subject = training.image_sampler.ImageSampler(
            image_sources=(MockImageSource(range(0, 10)),),
            default_validation_fraction=0.3, loop=True
        )
        subject.shuffle()
        order1 = [subject.get(idx) for idx in range(subject.num_training)]
        subject.shuffle()
        order2 = [subject.get(idx) for idx in range(subject.num_training)]
        self.assertNotEqual(order1, order2)

    def test_begin_calls_begin_on_inner_image_sources(self):
        image_source_1 = mock.create_autospec(MockImageSource(range(0, 10)))
        image_source_2 = mock.create_autospec(MockImageSource(range(10, 20)))
        image_source_3 = mock.create_autospec(MockImageSource(range(20, 30)))
        subject = training.image_sampler.ImageSampler((image_source_1, image_source_2, image_source_3),
                                                      default_validation_fraction=0.3)
        self.assertFalse(image_source_1.begin.called)
        self.assertFalse(image_source_2.begin.called)
        self.assertFalse(image_source_3.begin.called)
        subject.begin()
        self.assertTrue(image_source_1.begin.called)
        self.assertTrue(image_source_2.begin.called)
        self.assertTrue(image_source_3.begin.called)
