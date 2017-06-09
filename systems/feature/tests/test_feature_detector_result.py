import unittest
import numpy as np
import cv2
import bson.objectid
import operator
import database.tests.test_entity
import util.dict_utils as du
import systems.feature.feature_detector_result as feature_result


class TestSLAMTrialResult(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return feature_result.FeatureDetectorResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'image_source_id': np.random.randint(0, 10),
            'system_id': np.random.randint(10, 20),
            'keypoints': {
                bson.objectid.ObjectId(): [
                    cv2.KeyPoint(
                        x=np.random.uniform(0, 128),
                        y=np.random.uniform(0, 128),
                        _angle = np.random.uniform(360),
                        _class_id = np.random.randint(10),
                         _octave = np.random.randint(100000000),
                         _response = np.random.uniform(0, 1),
                         _size = np.random.uniform(10)
                    )
                    for _ in range(np.random.randint(50))]
                for _ in range(100)
            },
            'system_settings': {
                'a': np.random.randint(20, 30)
            }
        })
        return feature_result.FeatureDetectorResult(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two feature detector trial results models are equal
        :param trial_result1:
        :param trial_result2:
        :return:
        """
        if (not isinstance(trial_result1, feature_result.FeatureDetectorResult) or
                not isinstance(trial_result2, feature_result.FeatureDetectorResult)):
            self.fail('object was not a FeatureDetectorResult')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1.image_source_id, trial_result2.image_source_id)
        self.assertEqual(trial_result1.system_id, trial_result2.system_id)
        self.assertEqual(trial_result1.success, trial_result2.success)
        # Automatic comparison of this dict is extraordinarily slow, we have to unpack it
        self.assertEqual(len(trial_result1.keypoints), len(trial_result2.keypoints))
        self.assertEqual(set(trial_result1.keypoints.keys()), set(trial_result2.keypoints.keys()))
        for key in trial_result1.keypoints.keys():
            self.assertEqual(len(trial_result1.keypoints[key]), len(trial_result2.keypoints[key]))
            # we have to compare keypoints explicitly, because the == operator was not overloaded for them.
            points1 = sorted(trial_result1.keypoints[key], key=operator.attrgetter('response'))
            points2 = sorted(trial_result2.keypoints[key], key=operator.attrgetter('response'))
            for idx in range(len(points1)):
                self.assertEqual(points1[idx].pt, points2[idx].pt)
                self.assertEqual(points1[idx].angle, points2[idx].angle)
                self.assertEqual(points1[idx].class_id, points2[idx].class_id)
                self.assertEqual(points1[idx].octave, points2[idx].octave)
                self.assertEqual(points1[idx].response, points2[idx].response)
                self.assertEqual(points1[idx].size, points2[idx].size)
        self.assertEqual(trial_result1.settings, trial_result2.settings)

    def test_identifier(self):
        trial_result = self.make_instance(id_=123)
        self.assertEqual(trial_result.identifier, 123)
