import unittest
import numpy as np
import database.tests.test_entity
import util.dict_utils as du
import systems.slam.orbslam2


class TestORBSLAM2(database.tests.test_entity.EntityContract, unittest.TestCase):
    def get_class(self):
        return systems.slam.orbslam2.ORBSLAM2

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'vocabulary_file': 'imafile-{0}'.format(np.random.randint(10, 20)),
            'settings': {
                'Camera': {
                    'fx': np.random.uniform(10, 1000),
                    'fy': np.random.uniform(10, 1000),
                    'cx': np.random.uniform(10, 1000),
                    'cy': np.random.uniform(10, 1000),

                    'k1': np.random.uniform(-1, 1),
                    'k2': np.random.uniform(-1, 2),
                    'p1': np.random.uniform(0, 1),
                    'p2': np.random.uniform(0, 1),
                    'k3': np.random.uniform(-10, 10),
                    'fps': np.random.uniform(10, 100),
                    'RGB': np.random.randint(0, 1),
                },
                'ORBextractor': {
                    'nFeatures': np.random.randint(0, 8000),
                    'scaleFactor': np.random.uniform(0, 2),
                    'nLevels': np.random.randint(1, 10),
                    'iniThFAST': np.random.randint(0, 100),
                    'minThFAST': np.random.randint(0, 20)
                },
                'Viewer': {
                    'KeyFrameSize': np.random.uniform(0, 1),
                    'KeyFrameLineWidth': np.random.uniform(0, 3),
                    'GraphLineWidth': np.random.uniform(0, 3),
                    'PointSize': np.random.uniform(0, 3),
                    'CameraSize': np.random.uniform(0, 1),
                    'CameraLineWidth': np.random.uniform(0, 3),
                    'ViewpointX': np.random.uniform(0, 10),
                    'ViewpointY': np.random.uniform(0, 10),
                    'ViewpointZ': np.random.uniform(0, 10),
                    'ViewpointF': np.random.uniform(0, 10)
                }
            },
            'temp_folder': 'imafolder-{0}'.format(np.random.randint(10, 20))
        })
        return systems.slam.orbslam2.ORBSLAM2(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param trial_result1:
        :param trial_result2:
        :return:
        """
        if (not isinstance(trial_result1, systems.slam.orbslam2.ORBSLAM2) or
                not isinstance(trial_result2, systems.slam.orbslam2.ORBSLAM2)):
            self.fail('object was not a ORBSLAM2')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1._vocabulary_file, trial_result2._vocabulary_file)
        self.assertEqual(trial_result1._orbslam_settings, trial_result2._orbslam_settings)
        self.assertEqual(trial_result1.get_settings(), trial_result2.get_settings())
