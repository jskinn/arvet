import operator
import unittest

import bson.objectid
import cv2
import numpy as np

import core.sequence_type
import database.tests.test_entity
import trials.feature_detection.feature_detector_result as feature_result
import util.dict_utils as du
import util.transform as tf


class TestFeatureDetectorResult(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return feature_result.FeatureDetectorResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'system_id': np.random.randint(10, 20),
            'keypoints': {
                bson.objectid.ObjectId(): [
                    cv2.KeyPoint(
                        x=np.random.uniform(0, 128),
                        y=np.random.uniform(0, 128),
                        _angle=np.random.uniform(360),
                        _class_id=np.random.randint(10),
                        _octave=np.random.randint(100000000),
                        _response=np.random.uniform(0, 1),
                        _size=np.random.uniform(10)
                    )
                    for _ in range(np.random.randint(3))]
                for _ in range(10)
            },
            'sequence_type': core.sequence_type.ImageSequenceType.SEQUENTIAL,
            'system_settings': {
                'a': np.random.randint(20, 30)
            }
        })
        if 'timestamps' not in kwargs:
            kwargs['timestamps'] = {idx + np.random.uniform(0, 1): identifier
                                    for idx, identifier in enumerate(kwargs['keypoints'].keys())}
        if 'camera_poses' not in kwargs:
            kwargs['camera_poses'] = {identifier: tf.Transform(location=np.random.uniform(-1000, 1000, 3),
                                                               rotation=np.random.uniform(-1, 1, 4))
                                      for identifier in kwargs['keypoints'].keys()}
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
        self.assertEqual(trial_result1.system_id, trial_result2.system_id)
        self.assertEqual(trial_result1.success, trial_result2.success)
        self.assertEqual(trial_result1.sequence_type, trial_result2.sequence_type)
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
        self.assertEqual(set(trial_result1.timestamps.keys()), set(trial_result2.timestamps.keys()))
        for key in trial_result1.timestamps.keys():
            self.assertEqual(trial_result1.timestamps[key], trial_result2.timestamps[key])
        self.assertEqual(set(trial_result1.camera_poses.keys()), set(trial_result2.camera_poses.keys()))
        for key in trial_result1.camera_poses.keys():
            self.assertEqual(trial_result1.camera_poses[key], trial_result2.camera_poses[key])
        self.assertEqual(trial_result1.settings, trial_result2.settings)

    def assert_serialized_equal(self, s_model1, s_model2):
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in {'_id', '_type', 'keypoints', 'sequence_type', 'settings', 'settings'}:
            self.assertEqual(s_model1[key], s_model2[key])
        s_model1_stamps = {stamp: bson.objectid.ObjectId(identifier) for stamp, identifier in s_model1['timestamps']}
        s_model2_stamps = {stamp: bson.objectid.ObjectId(identifier) for stamp, identifier in s_model1['timestamps']}
        self.assertEqual(s_model1_stamps, s_model2_stamps)
        s_model1_poses = {stamp: tf.Transform.deserialize(s_trans) for stamp, s_trans in s_model1['camera_poses']}
        s_model2_poses = {stamp: tf.Transform.deserialize(s_trans) for stamp, s_trans in s_model2['camera_poses']}
        self.assertEqual(s_model1_poses, s_model2_poses)
