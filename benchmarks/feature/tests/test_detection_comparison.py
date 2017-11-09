# Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import unittest.mock as mock
import pickle
import bson
import gridfs
import cv2
import util.dict_utils as du
import util.transform as tf
import database.tests.test_entity as entity_test
import core.sequence_type
import core.trial_comparison
import trials.feature_detection.feature_detector_result as detector_result
import benchmarks.feature.detection_comparison as detection_comp


class TestFeatureDetectionComparison(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return detection_comp.FeatureDetectionComparison

    def make_instance(self, *args, **kwargs):
        return detection_comp.FeatureDetectionComparison(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: FeatureDetectionComparison
        :param benchmark2: FeatureDetectionComparison
        :return:
        """
        if (not isinstance(benchmark1, detection_comp.FeatureDetectionComparison) or
                not isinstance(benchmark2, detection_comp.FeatureDetectionComparison)):
            self.fail('object was not a FeatureDetectionComparison')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1._acceptable_radius, benchmark2._acceptable_radius)

    def test_comparison_returns_comparison_result(self):
        random = np.random.RandomState(1687)
        image_id = bson.ObjectId()
        trial_result1 = detector_result.FeatureDetectorResult(
            system_id=bson.ObjectId(),
            keypoints={image_id: [cv2.KeyPoint(x=p[0], y=p[1], _size=1) for p in random.randint(0, 300, (10, 2))]},
            timestamps={10: image_id},
            camera_poses={image_id: tf.Transform()},
            sequence_type=core.sequence_type.ImageSequenceType.NON_SEQUENTIAL,
            system_settings={}
        )
        trial_result2 = detector_result.FeatureDetectorResult(
            system_id=bson.ObjectId(),
            keypoints={image_id: [cv2.KeyPoint(x=p[0], y=p[1], _size=1) for p in random.randint(0, 300, (10, 2))]},
            timestamps={10: image_id},
            camera_poses={image_id: tf.Transform()},
            sequence_type=core.sequence_type.ImageSequenceType.NON_SEQUENTIAL,
            system_settings={}
        )

        comparison_benchmark = detection_comp.FeatureDetectionComparison()
        comparison_result = comparison_benchmark.compare_trial_results(trial_result1, trial_result2)
        self.assertIsInstance(comparison_result, core.trial_comparison.TrialComparisonResult)
        self.assertEqual(comparison_benchmark.identifier, comparison_result.benchmark)
        self.assertEqual(trial_result1.identifier, comparison_result.trial_result)
        self.assertEqual(trial_result2.identifier, comparison_result.reference_trial_result)


class MockReadable:
    """
    A helper for mock gridfs.get to return, that has a 'read' method as expected.
    """
    def __init__(self, thing):
        self._thing_bytes = pickle.dumps(thing, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self):
        return self._thing_bytes


class TestFeatureDetectionComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.changes = [{
            'trial_image_id': bson.ObjectId(),
            'reference_image_id': bson.ObjectId(),
            'point_matches': [((p[0], p[1]), (p[2], p[3])) for p in np.random.randint(0, 300, (10, 4))],
            'new_trial_points': [(p[0], p[1]) for p in np.random.randint(0, 300, (10, 2))],
            'missing_reference_points': [(p[0], p[1]) for p in np.random.randint(0, 300, (10, 2))]
        } for _ in range(3)]

    def get_class(self):
        return detection_comp.FeatureDetectionComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': bson.ObjectId(),
            'trial_result_id': bson.ObjectId(),
            'reference_id': bson.ObjectId(),
            'feature_changes': self.changes,
            'changes_id': bson.ObjectId()
        })
        return detection_comp.FeatureDetectionComparisonResult(*args, **kwargs)

    def create_mock_db_client(self):
        self.db_client = super().create_mock_db_client()

        self.db_client.grid_fs = mock.create_autospec(gridfs.GridFS)
        self.db_client.grid_fs.get.return_value = MockReadable(self.changes)

        return self.db_client

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark1: FeatureDetectionComparisonResult
        :param benchmark2: FeatureDetectionComparisonResult
        :return:
        """
        if (not isinstance(benchmark1, detection_comp.FeatureDetectionComparisonResult) or
                not isinstance(benchmark2, detection_comp.FeatureDetectionComparisonResult)):
            self.fail('object was not a FeatureDetectionComparisonResult')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.success, benchmark2.success)
        self.assertEqual(benchmark1.benchmark, benchmark2.benchmark)
        self.assertEqual(benchmark1.trial_result, benchmark2.trial_result)
        self.assertEqual(benchmark1.reference_trial_result, benchmark2.reference_trial_result)
        self.assertEqual(benchmark1._feature_changes, benchmark2._feature_changes)
        self.assertEqual(benchmark1._changes_id, benchmark2._changes_id)

    def test_save_data_stores_keypoints(self):
        mock_db_client = self.create_mock_db_client()
        new_id = bson.ObjectId()
        mock_db_client.grid_fs.put.return_value = new_id

        subject = self.make_instance(changes_id=None)
        subject.save_data(mock_db_client)
        self.assertTrue(mock_db_client.grid_fs.put.called)
        stored_changes = pickle.loads(mock_db_client.grid_fs.put.call_args[0][0])
        self.assertEqual(self.changes, stored_changes)
