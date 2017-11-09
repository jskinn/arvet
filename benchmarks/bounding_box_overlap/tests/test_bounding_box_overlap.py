# Copyright (c) 2017, John Skinner
import unittest
import bson.objectid as oid
import database.tests.test_entity
import core.benchmark
import trials.object_detection.bounding_box_result as bbox_trial
import benchmarks.bounding_box_overlap.bounding_box_overlap as bbox_overlap
import benchmarks.bounding_box_overlap.bounding_box_overlap_result as bbox_result


class MockTrialResult:

    def __init__(self, gt_bboxes, bboxes):
        self._gt_bboxes = gt_bboxes
        self._bboxes = bboxes

    @property
    def identifier(self):
        return 'ThisIsAMockTrialResult'

    def get_ground_truth_bounding_boxes(self):
        return self._gt_bboxes

    def get_bounding_boxes(self):
        return self._bboxes


class TestBoundingBoxOverlapBenchmark(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return bbox_overlap.BoundingBoxOverlapBenchmark

    def make_instance(self, *args, **kwargs):
        return bbox_overlap.BoundingBoxOverlapBenchmark(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: BenchmarkLoopClosure
        :param benchmark2: BenchmarkLoopClosure
        :return:
        """
        if (not isinstance(benchmark1, bbox_overlap.BoundingBoxOverlapBenchmark) or
                not isinstance(benchmark2, bbox_overlap.BoundingBoxOverlapBenchmark)):
            self.fail('object was not a BoundingBoxOverlapBenchmark')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)

    def test_benchmark_results_returns_a_benchmark_result(self):
        trial_result = MockTrialResult(
            gt_bboxes={oid.ObjectId(): [bbox_trial.BoundingBox(('cup',), 0.8256, 15, 22, 100, 100)]},
            bboxes={oid.ObjectId(): [bbox_trial.BoundingBox(('cup',), 0.8256, 15, 22, 100, 100)]})

        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)
        self.assertIsInstance(result, core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertIsInstance(result, bbox_result.BoundingBoxOverlapBenchmarkResult)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(trial_result.identifier, result.trial_result)

    def test_benchmark_measures_score_per_gt_bounding_box(self):
        id1 = oid.ObjectId()
        id2 = oid.ObjectId()
        id3 = oid.ObjectId()
        id4 = oid.ObjectId()
        trial_result = MockTrialResult(
            gt_bboxes={
                id1: [bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 100, 100)],
                id2: [bbox_trial.BoundingBox({'car'}, 1, 15, 22, 100, 100)],
                id3: [bbox_trial.BoundingBox({'cow'}, 1, 15, 22, 100, 100)],
                id4: [bbox_trial.BoundingBox({'cat'}, 1, 15, 22, 100, 100)]
            },
            bboxes={
                id1: [bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 100, 100)],       # Matched exactly
                id2: [bbox_trial.BoundingBox({'car'}, 0.8256, 15, 22, 100, 100)],  # Only confidence reduced
                id3: [bbox_trial.BoundingBox({'cow'}, 1, 25, 32, 95, 95)],         # Slightly misplaced
                id4: [bbox_trial.BoundingBox({'cat'}, 0.75, 25, 32, 95, 95)]       # Reduced confidence and slightly misplaced
            }
        )
        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)

        self.assertIn(id1, result.overlaps)
        self.assertIn(id2, result.overlaps)
        self.assertIn(id3, result.overlaps)
        self.assertIn(id4, result.overlaps)
        self.assertEqual(1, len(result.overlaps[id1]))
        self.assertEqual(1, len(result.overlaps[id2]))
        self.assertEqual(1, len(result.overlaps[id3]))
        self.assertEqual(1, len(result.overlaps[id4]))
        self.assertEqual({
            'overlap': 10000,
            'bounding_box_area': 10000,
            'ground_truth_area': 10000,
            'confidence': 1.0,
            'bounding_box_classes': ('cup',),
            'ground_truth_classes': ('cup',)
        }, result.overlaps[id1][0])
        self.assertEqual({
            'overlap': 10000,
            'bounding_box_area': 10000,
            'ground_truth_area': 10000,
            'confidence': 0.8256,
            'bounding_box_classes': ('car',),
            'ground_truth_classes': ('car',)
        }, result.overlaps[id2][0])
        self.assertEqual({
            'overlap': 8100,
            'bounding_box_area': 9025,
            'ground_truth_area': 10000,
            'confidence': 1.0,
            'bounding_box_classes': ('cow',),
            'ground_truth_classes': ('cow',)
        }, result.overlaps[id3][0])
        self.assertEqual({
            'overlap': 8100,
            'bounding_box_area': 9025,
            'ground_truth_area': 10000,
            'confidence': 0.75,
            'bounding_box_classes': ('cat',),
            'ground_truth_classes': ('cat',)
        }, result.overlaps[id4][0])

    def test_benchmark_selects_highest_overlap_bounding_box(self):
        id1 = oid.ObjectId()
        trial_result = MockTrialResult(
            gt_bboxes={
                id1: [bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 100, 100),
                      bbox_trial.BoundingBox({'cup'}, 1, 115, 122, 50, 50),
                      bbox_trial.BoundingBox({'cup'}, 1, 165, 172, 25, 25)]
            },
            bboxes={
                id1: [bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)],
            }
        )
        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)

        self.assertIn(id1, result.overlaps)
        self.assertEqual(3, len(result.overlaps[id1]))
        self.assertEqual({
            'overlap': 10000,
            'bounding_box_area': 30625,
            'ground_truth_area': 10000,
            'confidence': 1.0,
            'bounding_box_classes': ('cup',),
            'ground_truth_classes': ('cup',)
        }, result.overlaps[id1][0])
        self.assertEqual({
            'overlap': 0,
            'bounding_box_area': 0,
            'ground_truth_area': 2500,
            'confidence': 0,
            'bounding_box_classes': tuple(),
            'ground_truth_classes': ('cup',)
        }, result.overlaps[id1][1])
        self.assertEqual({
            'overlap': 0,
            'bounding_box_area': 0,
            'ground_truth_area': 625,
            'confidence': 0,
            'bounding_box_classes': tuple(),
            'ground_truth_classes': ('cup',)
        }, result.overlaps[id1][2])

    def test_benchmark_matches_each_gt_box_only_once(self):
        id1 = oid.ObjectId()
        trial_result = MockTrialResult(
            gt_bboxes={
                id1: [bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)]
            },
            bboxes={
                id1: [bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 100, 100),
                      bbox_trial.BoundingBox({'cup'}, 1, 115, 122, 50, 50),
                      bbox_trial.BoundingBox({'cup'}, 1, 165, 172, 25, 25)],
            }
        )
        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)

        self.assertIn(id1, result.overlaps)
        self.assertEqual(3, len(result.overlaps[id1]))
        self.assertEqual({
            'overlap': 10000,
            'bounding_box_area': 10000,
            'ground_truth_area': 30625,
            'confidence': 1.0,
            'bounding_box_classes': ('cup',),
            'ground_truth_classes': ('cup',)
        }, result.overlaps[id1][0])
        self.assertEqual({
            'overlap': 0,
            'bounding_box_area': 2500,
            'ground_truth_area': 0,
            'confidence': 1.0,
            'bounding_box_classes': ('cup',),
            'ground_truth_classes': tuple()
        }, result.overlaps[id1][1])
        self.assertEqual({
            'overlap': 0,
            'bounding_box_area': 625,
            'ground_truth_area': 0,
            'confidence': 1.0,
            'bounding_box_classes': ('cup',),
            'ground_truth_classes': tuple()
        }, result.overlaps[id1][2])

    def test_overlap_computes_bbox_overlap(self):
        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        self.assertEqual(30625, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 25, 32, 175, 175)
        self.assertEqual(27225, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 5, 12, 175, 175)
        self.assertEqual(27225, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 25, 32, 100, 100)
        self.assertEqual(10000, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 190, 22, 175, 175)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 15, 197, 175, 175)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 190, 197, 175, 175)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 190, 22, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 197, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 190, 197, 175, 175)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 175, 175)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 190, 197, 10, 10)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 10, 10)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))

        bbox1 = bbox_trial.BoundingBox({'cup'}, 1, 15, 22, 10, 10)
        bbox2 = bbox_trial.BoundingBox({'cup'}, 1, 190, 197, 10, 10)
        self.assertEqual(0, bbox_overlap.compute_overlap(bbox1, bbox2))
