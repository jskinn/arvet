import unittest
import unittest.mock as mock
import abc
import bson.objectid
import database.tests.test_entity
import util.dict_utils as du
import core.trained_system


class TrainerContract(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def make_instance(self, *args, **kwargs):
        """
        Make an instance of the system under test, with some default arguments
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def make_mock_trainee(self):
        """
        Create a mock trainee.
        Subclasses should override this to make sure that the mock trainee
        can be trained by the trainer.
        :return: An instance of VisionSystemTrainee
        """
        return mock.create_autospec(core.trained_system.VisionSystemTrainee)

    def test_calls_trainee_in_correct_order(self):
        mock_trainee = self.make_mock_trainee()
        subject = self.make_instance()
        subject.train_vision_system(mock_trainee)

        # Make sure each of the required methods is called at least once
        self.assertTrue(mock_trainee.start_training.called)
        self.assertTrue(mock_trainee.train_with_image.called)
        self.assertTrue(mock_trainee.finish_training.called)
        if mock_trainee.start_validation.called:
            self.assertTrue(mock_trainee.validate_with_image.called)
            self.assertTrue(mock_trainee.finish_validation.called)

        is_training = False
        open_training = 0
        is_validating = False
        open_validating = 0
        for call in mock_trainee.method_calls:
            if is_training:
                self.assertNotEqual('start_training', call[0])  # No nested calls to start_training

                if is_validating:
                    self.assertNotEqual('start_validation', call[0])    # We're already validating, don't start
                    self.assertNotEqual('train_with_image', call[0])    # don't train while validating
                    self.assertNotEqual('finish_training', call[0])     # We're validating, don't stop training
                    if call[0] == 'finish_validation':
                        is_validating = False
                        open_validating -= 1
                else:
                    self.assertNotEqual('validate_with_image', call[0])     # Not validating, don't give validation images
                    self.assertNotEqual('finish_validation', call[0])       # Can't finish without starting
                    if call[0] == 'start_validation':
                        is_validating = True
                        open_validating += 1

                if call[0] == 'finish_training':
                    is_training = False
                    open_training -= 1
            else:
                self.assertNotEqual('train_with_image', call[0])    # Can't train_with_image, since we're not training
                self.assertNotEqual('finish_training', call[0])     # Can't finish without starting
                self.assertNotEqual('start_validation', call[0])    # No validation when not training
                self.assertNotEqual('validate_with_image', call[0])
                self.assertNotEqual('finish_validation', call[0])
                if call[0] == 'start_training':
                    is_training = True
                    open_training += 1
        self.assertFalse(is_training)
        self.assertEqual(0, open_training)  # Make sure we start and finish training the same number of times
        self.assertFalse(is_validating)
        self.assertEqual(0, open_validating)  # Make sure we start and finish validating the same number of times


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
            ],
            'training_settings': {
                'A': 'Yes',
                'but really?': True
            }
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
        self.assertEqual(trial_result1.training_settings, trial_result2.training_settings)
