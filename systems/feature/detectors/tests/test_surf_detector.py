import unittest
import util.dict_utils as du
import database.tests.test_entity
import systems.feature.detectors.surf_detector as surf_detector
import systems.feature.tests.test_feature_detector


class TestSiftDetector(database.tests.test_entity.EntityContract,
                       systems.feature.tests.test_feature_detector.FeatureDetectorContract, unittest.TestCase):

    def get_class(self):
        return surf_detector.SurfDetector

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'config': {
                'hessian_threshold': 122.4,
                'num_octaves': 3,
                'num_octave_layers': 4,
                'extended': True,
                'upright': False
            }
        })
        return surf_detector.SurfDetector(*args, **kwargs)

    def get_config_attributes(self):
        return ['hessian_threshold', 'num_octaves', 'num_octave_layers', 'extended', 'upright']

    def assert_models_equal(self, detector1, detector2):
        """
        Helper to assert that two detectors detectors are equal
        :param detector1: SiftDetector
        :param detector2: SiftDetector
        :return:
        """
        if (not isinstance(detector1, surf_detector.SurfDetector)
                or not isinstance(detector2, surf_detector.SurfDetector)):
            self.fail('object was not an SiftDetector')
        self.assertEqual(detector1.identifier, detector2.identifier)
        self.assertEqual(detector1.hessian_threshold, detector2.hessian_threshold)
        self.assertEqual(detector1.num_octaves, detector2.num_octaves)
        self.assertEqual(detector1.num_octave_layers, detector2.num_octave_layers)
        self.assertEqual(detector1.extended, detector2.extended)
        self.assertEqual(detector1.upright, detector2.upright)

    def test_get_system_settings(self):
        subject = self.make_instance()
        settings = subject.get_system_settings()
        self.assertEqual(122.4, settings['hessian_threshold'])
        self.assertEqual(3, settings['num_octaves'])
        self.assertEqual(4, settings['num_octave_layers'])
        self.assertEqual(True, settings['extended'])
        self.assertEqual(False, settings['upright'])
