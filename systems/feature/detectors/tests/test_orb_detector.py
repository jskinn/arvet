import unittest
import util.dict_utils as du
import database.tests.test_entity
import systems.feature.detectors.orb_detector as orb_detector
import systems.feature.tests.test_feature_detector


class TestSiftDetector(database.tests.test_entity.EntityContract,
                       systems.feature.tests.test_feature_detector.FeatureDetectorContract, unittest.TestCase):

    def get_class(self):
        return orb_detector.ORBDetector

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'config': {
                'num_features': 1985,
                'scale_factor': 1.3,
                'num_levels': 11,
                'edge_threshold': 22,
                'patch_size': 21,
                'fast_threshold': 19
            }
        })
        return orb_detector.ORBDetector(*args, **kwargs)

    def get_config_attributes(self):
        return ['num_features', 'num_octave_layers', 'contrast_threshold', 'edge_threshold', 'sigma']

    def assert_models_equal(self, detector1, detector2):
        """
        Helper to assert that two detectors detectors are equal
        :param detector1: ORBDetector
        :param detector2: ORBDetector
        :return:
        """
        if (not isinstance(detector1, orb_detector.ORBDetector)
                or not isinstance(detector2, orb_detector.ORBDetector)):
            self.fail('object was not an SiftDetector')
        self.assertEqual(detector1.identifier, detector2.identifier)
        self.assertEqual(detector1.num_features, detector2.num_features)
        self.assertEqual(detector1.scale_factor, detector2.scale_factor)
        self.assertEqual(detector1.number_of_levels, detector2.number_of_levels)
        self.assertEqual(detector1.patch_size, detector2.patch_size)
        self.assertEqual(detector1.edge_threshold, detector2.edge_threshold)
        self.assertEqual(detector1.fast_threshold, detector2.fast_threshold)

    def test_get_system_settings(self):
        subject = self.make_instance()
        settings = subject.get_system_settings()
        self.assertEqual(1985, settings['num_features'])
        self.assertEqual(1.3, settings['scale_factor'])
        self.assertEqual(11, settings['num_levels'])
        self.assertEqual(22, settings['edge_threshold'])
        self.assertEqual(21, settings['patch_size'])
        self.assertEqual(19, settings['fast_threshold'])
