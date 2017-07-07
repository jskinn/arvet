import unittest
import bson.objectid
import database.tests.test_entity
import util.dict_utils as du
import core.trained_system


class MockTrainedSystem(core.trained_system.TrainedVisionSystem):
    """
    A simple mock for the trained system, with trivial implementations of the abstract methods.
    They're not what is under test here anyway.
    """

    @property
    def is_deterministic(self):
        return True

    def is_image_source_appropriate(self, image_source):
        return True

    def start_trial(self, sequence_type):
        pass

    def process_image(self, image, timestamp):
        pass

    def finish_trial(self):
        return None


class TestTrainedSystem(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return MockTrainedSystem

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'vision_system_trainer': bson.objectid.ObjectId(),
            'training_image_source_ids': [
                bson.objectid.ObjectId(),
                bson.objectid.ObjectId(),
                bson.objectid.ObjectId(),
                bson.objectid.ObjectId()
            ]
        })
        return MockTrainedSystem(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two dataset models are equal
        :param trial_result1: Dataset
        :param trial_result2: Dataset
        :return:
        """
        if (not isinstance(trial_result1, core.trained_system.TrainedVisionSystem) or
                not isinstance(trial_result2, core.trained_system.TrainedVisionSystem)):
            self.fail('object was not a TrainedVisionSystem')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1.vision_system_trainer, trial_result2.vision_system_trainer)
        self.assertEqual(trial_result1.training_image_sources, trial_result2.training_image_sources)
