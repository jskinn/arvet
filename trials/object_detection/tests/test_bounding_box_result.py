import operator
import unittest

import bson.objectid
import numpy as np

import database.tests.test_entity
import metadata.image_metadata as imeta
import trials.object_detection.bounding_box_result as bbox_result
import util.dict_utils as du


class TestBoundingBoxResult(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return bbox_result.BoundingBoxResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'system_id': np.random.randint(10, 20),
            'bounding_boxes': {
                bson.objectid.ObjectId(): {
                    imeta.BoundingBox(
                        class_name='class_' + str(np.random.randint(255)),
                        confidence=np.random.uniform(0, 1),
                        x=np.random.randint(800),
                        y=np.random.randint(600),
                        width=np.random.randint(128),
                        height=np.random.randint(128)
                    )
                    for _ in range(np.random.randint(50))}
                for _ in range(100)
            },
            'ground_truth_bounding_boxes': {
                bson.objectid.ObjectId(): {
                    imeta.BoundingBox(
                        class_name='class_' + str(np.random.randint(255)),
                        confidence=np.random.uniform(0, 1),
                        x=np.random.randint(800),
                        y=np.random.randint(600),
                        width=np.random.randint(128),
                        height=np.random.randint(128)
                    )
                    for _ in range(np.random.randint(50))}
                for _ in range(100)
            },
            'system_settings': {
                'a': np.random.randint(20, 30)
            }
        })
        return bbox_result.BoundingBoxResult(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two feature detector trial results models are equal
        :param trial_result1:
        :param trial_result2:
        :return:
        """
        if (not isinstance(trial_result1,  bbox_result.BoundingBoxResult) or
                not isinstance(trial_result2, bbox_result.BoundingBoxResult)):
            self.fail('object was not a BoundingBoxResult')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1.system_id, trial_result2.system_id)
        self.assertEqual(trial_result1.success, trial_result2.success)
        # Automatic comparison of this dict is extraordinarily slow, we have to unpack it
        self._assert_bboxes_equal(trial_result1.bounding_boxes, trial_result2.bounding_boxes)
        self._assert_bboxes_equal(trial_result1.ground_truth_bounding_boxes, trial_result2.ground_truth_bounding_boxes)
        self.assertEqual(trial_result1.settings, trial_result2.settings)

    def assert_serialized_equal(self, s_model1, s_model2):
        self.assertEqual(len(s_model1), len(s_model2))
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'bounding_boxes' and key is not 'gt_bounding_boxes':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special comparison for the bounding boxes, because they're sets, so we don't care about order
        for bbox_key in ('bounding_boxes', 'gt_bounding_boxes'):
            self.assertEqual(set(s_model1[bbox_key].keys()), set(s_model2[bbox_key].keys()))
            for key in s_model1[bbox_key].keys():
                bboxes1 = {imeta.BoundingBox.deserialize(s_bbox) for s_bbox in s_model1[bbox_key][key]}
                bboxes2 = {imeta.BoundingBox.deserialize(s_bbox) for s_bbox in s_model2[bbox_key][key]}
                self.assertEqual(bboxes1, bboxes2)

    def _assert_bboxes_equal(self, bboxes1, bboxes2):
        self.assertEqual(len(bboxes1), len(bboxes2))
        self.assertEqual(set(bboxes1.keys()), set(bboxes2.keys()))
        for key in bboxes1.keys():
            self.assertEqual(bboxes1[key], bboxes2[key])

    def test_identifier(self):
        trial_result = self.make_instance(id_=123)
        self.assertEqual(trial_result.identifier, 123)
