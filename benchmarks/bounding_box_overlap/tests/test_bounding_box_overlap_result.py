import unittest
import bson.objectid
import numpy as np

import database.tests.test_entity
import util.dict_utils as du
import benchmarks.bounding_box_overlap.bounding_box_overlap_result as overlap_result


class TestBoundingBoxOverlapBenchmarkResult(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return overlap_result.BoundingBoxOverlapBenchmarkResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(10, 20),
            'trial_result_id': np.random.randint(20, 30),
            'overlaps': {
                bson.objectid.ObjectId(): [{
                    'overlap': np.random.randint(0, 100),
                    'bounding_box_area': np.random.randint(100, 150),
                    'ground_truth_area': np.random.randint(100, 150),
                    'confidence': np.random.uniform(0.2, 1),
                    'bounding_box_classes': ('class-' + str(idx)),
                    'ground_truth_classes': ('class-' + str(idx))
                } for idx in range(np.random.randint(10))]
                for _ in range(100)
            },
            'settings': {
                'a': np.random.randint(20, 30)
            }
        })
        return overlap_result.BoundingBoxOverlapBenchmarkResult(*args, **kwargs)

    def assert_models_equal(self, result1, result2):
        """
        Helper to assert that two bounding box benchmark results are equal
        :param result1:
        :param result2:
        :return:
        """
        if (not isinstance(result1, overlap_result.BoundingBoxOverlapBenchmarkResult) or
                not isinstance(result2, overlap_result.BoundingBoxOverlapBenchmarkResult)):
            self.fail('object was not a BoundingBoxOverlapBenchmarkResult')
        self.assertEqual(result1.identifier, result2.identifier)
        self.assertEqual(result1.benchmark, result2.benchmark)
        self.assertEqual(result1.trial_result, result2.trial_result)
        self.assertEqual(result1.success, result2.success)
        self.assertEqual(result1.settings, result2.settings)
        self.assertEqual(set(result1.overlaps.keys()), set(result2.overlaps.keys()))
        for img_id in result1.overlaps.keys():
            self.assertEqual(len(result1.overlaps[img_id]), len(result2.overlaps[img_id]))
            for idx in range(len(result1.overlaps[img_id])):
                self.assertEqual(result1.overlaps[img_id][idx], result2.overlaps[img_id][idx])

    def assert_serialized_equal(self, s_model1, s_model2):
        self.assertEqual(len(s_model1), len(s_model2))
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'overlaps':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special comparison for the bounding boxes, because they're sets, so we don't care about order
        self.assertEqual(set(s_model1['overlaps'].keys()), set(s_model2['overlaps'].keys()))
        for img_id in s_model1['overlaps'].keys():
            self.assertEqual(len(s_model1['overlaps'][img_id]), len(s_model2['overlaps'][img_id]))
            for idx in range(len(s_model1['overlaps'][img_id])):
                self.assertEqual(s_model1['overlaps'][img_id][idx], s_model2['overlaps'][img_id][idx])

    def test_identifier(self):
        trial_result = self.make_instance(id_=123)
        self.assertEqual(trial_result.identifier, 123)

    def test_map_results(self):
        subject = overlap_result.BoundingBoxOverlapBenchmarkResult(
            benchmark_id=np.random.randint(10, 20),
            trial_result_id=np.random.randint(20, 30),
            settings={},
            overlaps={
                bson.objectid.ObjectId(): [{
                    'overlap': 22,
                    'bounding_box_area': np.random.randint(100, 150),
                    'ground_truth_area': np.random.randint(100, 150),
                    'confidence': np.random.uniform(0.2, 1),
                    'bounding_box_classes': ('cup', 'car'),
                    'ground_truth_classes': ('cup', 'class-' + str(np.random.randint(100)))
                }, {
                    'overlap': 41,
                    'bounding_box_area': np.random.randint(100, 150),
                    'ground_truth_area': np.random.randint(100, 150),
                    'confidence': np.random.uniform(0.2, 1),
                    'bounding_box_classes': ('cat',),
                    'ground_truth_classes': ('cat', 'class-' + str(np.random.randint(100)))
                }],
                bson.objectid.ObjectId(): [{
                    'overlap': 56,
                    'bounding_box_area': np.random.randint(100, 150),
                    'ground_truth_area': np.random.randint(100, 150),
                    'confidence': np.random.uniform(0.2, 1),
                    'bounding_box_classes': ('cat', 'car'),
                    'ground_truth_classes': ('cat', 'class-' + str(np.random.randint(100)))
                }]
            })
        mapped = subject.map_results(lambda x: x['bounding_box_classes'], lambda x: x['overlap'])
        self.assertEqual({'cat', 'cup', 'car'}, set(mapped.keys()))
        self.assertEqual([22], mapped['cup'])
        self.assertIn(22, mapped['car'])
        self.assertIn(56, mapped['car'])
        self.assertEqual(2, len(mapped['car']))
        self.assertIn(41, mapped['cat'])
        self.assertIn(56, mapped['cat'])
        self.assertEqual(2, len(mapped['cat']))
