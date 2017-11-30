# Copyright (c) 2017, John Skinner
import operator
import unittest
import unittest.mock as mock

import bson
import gridfs
import pickle
import cv2
import numpy as np

import argus.core.sequence_type
import argus.database.tests.test_entity
import argus.trials.feature_detection.feature_detector_result as feature_result
import argus.util.dict_utils as du
import argus.util.transform as tf


class MockReadable:
    """
    A helper for mock gridfs.get to return, that has a 'read' method as expected.
    """
    def __init__(self, thing):
        self._thing_bytes = pickle.dumps(thing, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self):
        return self._thing_bytes


class TestFeatureDetectorResult(argus.database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.keypoints = {
            bson.ObjectId(): [
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
        }
        self.s_keypoints = {identifier: [feature_result.serialize_keypoint(keypoint) for keypoint in keypoints]
                            for identifier, keypoints in self.keypoints.items()}

    def get_class(self):
        return feature_result.FeatureDetectorResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'system_id': np.random.randint(10, 20),
            'keypoints': self.keypoints,
            'sequence_type': argus.core.sequence_type.ImageSequenceType.SEQUENTIAL,
            'system_settings': {
                'a': np.random.randint(20, 30)
            },
            'keypoints_id': bson.ObjectId()
        })
        if 'timestamps' not in kwargs:
            kwargs['timestamps'] = {idx + np.random.uniform(0, 1): identifier
                                    for idx, identifier in enumerate(kwargs['keypoints'].keys())}
        if 'camera_poses' not in kwargs:
            kwargs['camera_poses'] = {identifier: tf.Transform(location=np.random.uniform(-1000, 1000, 3),
                                                               rotation=np.random.uniform(-1, 1, 4))
                                      for identifier in kwargs['keypoints'].keys()}
        return feature_result.FeatureDetectorResult(*args, **kwargs)

    def create_mock_db_client(self):
        self.db_client = super().create_mock_db_client()

        self.db_client.grid_fs = unittest.mock.create_autospec(gridfs.GridFS)
        self.db_client.grid_fs.get.return_value = MockReadable(self.s_keypoints)

        return self.db_client

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
        for key in {'_id', '_type', 'keypoints_id', 'sequence_type', 'settings', 'settings'}:
            self.assertEqual(s_model1[key], s_model2[key])
        s_model1_stamps = {stamp: bson.objectid.ObjectId(identifier) for stamp, identifier in s_model1['timestamps']}
        s_model2_stamps = {stamp: bson.objectid.ObjectId(identifier) for stamp, identifier in s_model1['timestamps']}
        self.assertEqual(s_model1_stamps, s_model2_stamps)
        s_model1_poses = {stamp: tf.Transform.deserialize(s_trans) for stamp, s_trans in s_model1['camera_poses']}
        s_model2_poses = {stamp: tf.Transform.deserialize(s_trans) for stamp, s_trans in s_model2['camera_poses']}
        self.assertEqual(s_model1_poses, s_model2_poses)

    def test_save_data_stores_keypoints(self):
        mock_db_client = self.create_mock_db_client()
        new_id = bson.ObjectId()
        mock_db_client.grid_fs.put.return_value = new_id

        subject = self.make_instance(keypoints_id=None)
        subject.save_data(mock_db_client)
        self.assertTrue(mock_db_client.grid_fs.put.called)
        s_keypoints = pickle.loads(mock_db_client.grid_fs.put.call_args[0][0])
        self.assertEqual(self.s_keypoints, s_keypoints)

    def test_serialize_and_deserialize_keypoint(self):
        keypoint1 = cv2.KeyPoint(
            x=np.random.uniform(0, 128),
            y=np.random.uniform(0, 128),
            _angle=np.random.uniform(360),
            _class_id=np.random.randint(10),
            _octave=np.random.randint(100000000),
            _response=np.random.uniform(0, 1),
            _size=np.random.uniform(10)
        )
        s_keypoint1 = feature_result.serialize_keypoint(keypoint1)

        keypoint2 = feature_result.deserialize_keypoint(s_keypoint1)
        s_keypoint2 = feature_result.serialize_keypoint(keypoint2)

        self.assertEqual(keypoint1.pt, keypoint2.pt)
        self.assertEqual(keypoint1.angle, keypoint2.angle)
        self.assertEqual(keypoint1.class_id, keypoint2.class_id)
        self.assertEqual(keypoint1.octave, keypoint2.octave)
        self.assertEqual(keypoint1.response, keypoint2.response)
        self.assertEqual(keypoint1.size, keypoint2.size)
        self.assertEqual(s_keypoint1, s_keypoint2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            keypoint2 = feature_result.deserialize_keypoint(s_keypoint2)
            s_keypoint2 = feature_result.serialize_keypoint(keypoint2)
            self.assertEqual(keypoint1.pt, keypoint2.pt)
            self.assertEqual(keypoint1.angle, keypoint2.angle)
            self.assertEqual(keypoint1.class_id, keypoint2.class_id)
            self.assertEqual(keypoint1.octave, keypoint2.octave)
            self.assertEqual(keypoint1.response, keypoint2.response)
            self.assertEqual(keypoint1.size, keypoint2.size)
            self.assertEqual(s_keypoint1, s_keypoint2)
