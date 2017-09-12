#Copyright (c) 2017, John Skinner
import unittest
import util.dict_utils as du
import database.tests.test_entity
import systems.feature.detectors.sift_detector as sift_detector
import systems.feature.tests.test_feature_detector


class TestSiftDetector(database.tests.test_entity.EntityContract,
                       systems.feature.tests.test_feature_detector.FeatureDetectorContract, unittest.TestCase):

    def get_class(self):
        return sift_detector.SiftDetector

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'config': {
                'num_features': 0,
                'num_octave_layers': 4,
                'contrast_threshold': 0.04,
                'edge_threshold': 10,
                'sigma': 1.6
            }
        })
        return sift_detector.SiftDetector(*args, **kwargs)

    def get_config_attributes(self):
        return ['num_features', 'num_octave_layers', 'contrast_threshold', 'edge_threshold', 'sigma']

    def assert_models_equal(self, detector1, detector2):
        """
        Helper to assert that two detectors detectors are equal
        :param detector1: SiftDetector
        :param detector2: SiftDetector
        :return:
        """
        if (not isinstance(detector1, sift_detector.SiftDetector)
                or not isinstance(detector2, sift_detector.SiftDetector)):
            self.fail('object was not an SiftDetector')
        self.assertEqual(detector1.identifier, detector2.identifier)
        self.assertEqual(detector1.num_features, detector2.num_features)
        self.assertEqual(detector1.num_octave_layers, detector2.num_octave_layers)
        self.assertEqual(detector1.contrast_threshold, detector2.contrast_threshold)
        self.assertEqual(detector1.edge_threshold, detector2.edge_threshold)
        self.assertEqual(detector1.sigma, detector2.sigma)

    def test_get_system_settings(self):
        subject = self.make_instance()
        settings = subject.get_system_settings()
        self.assertEqual(0, settings['num_features'])
        self.assertEqual(4, settings['num_octave_layers'])
        self.assertEqual(0.04, settings['contrast_threshold'])
        self.assertEqual(10, settings['edge_threshold'])
        self.assertEqual(1.6, settings['sigma'])
