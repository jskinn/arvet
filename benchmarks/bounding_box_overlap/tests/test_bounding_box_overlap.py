import unittest
import bson.objectid as oid
import database.tests.test_entity
import core.benchmark
import metadata.image_metadata as imeta
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
            gt_bboxes={oid.ObjectId(): imeta.BoundingBox('cup', 0.8256, 15, 22, 100, 100)},
            bboxes={oid.ObjectId(): imeta.BoundingBox('cup', 0.8256, 15, 22, 100, 100)})

        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)
        self.assertIsInstance(result, core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertIsInstance(result, bbox_result.BoundingBoxOverlapBenchmarkResult)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(trial_result.identifier, result.trial_result)

    def test_benchmark_measures_score_by_class(self):
        id1 = oid.ObjectId()
        id2 = oid.ObjectId()
        id3 = oid.ObjectId()
        id4 = oid.ObjectId()
        trial_result = MockTrialResult(
            gt_bboxes={
                id1: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)],
                id2: [imeta.BoundingBox('car', 1, 15, 22, 100, 100)],
                id3: [imeta.BoundingBox('cow', 1, 15, 22, 100, 100)],
                id4: [imeta.BoundingBox('cat', 1, 15, 22, 100, 100)]
            },
            bboxes={
                id1: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)],       # Matched exactly
                id2: [imeta.BoundingBox('car', 0.8256, 15, 22, 100, 100)],  # Only confidence reduced
                id3: [imeta.BoundingBox('cow', 1, 25, 32, 95, 95)],         # Slightly misplaced
                id4: [imeta.BoundingBox('cat', 0.75, 25, 32, 95, 95)]       # Reduced confidence and slightly misplaced
            }
        )
        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)

        self.assertIn('cup', result.classes)
        self.assertIn('car', result.classes)
        self.assertIn('cow', result.classes)
        self.assertIn('cat', result.classes)
        self.assertEqual(1, len(result.scores_by_class['cup']))
        self.assertEqual(1, len(result.scores_by_class['car']))
        self.assertEqual(1, len(result.scores_by_class['cow']))
        self.assertEqual(1, len(result.scores_by_class['cat']))
        self.assertEqual(1, result.get_max_score('cup'))
        self.assertEqual(0.8256, result.get_max_score('car'))

    def test_benchmark_aggregates_score_by_class(self):
        id1 = oid.ObjectId()
        id2 = oid.ObjectId()
        id3 = oid.ObjectId()
        id4 = oid.ObjectId()
        trial_result = MockTrialResult(
            gt_bboxes={
                id1: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)],
                id2: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)],
                id3: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)],
                id4: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)]
            },
            bboxes={
                id1: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100)],       # Matched exactly
                id2: [imeta.BoundingBox('cup', 0.8256, 15, 22, 100, 100)],  # Only confidence reduced
                id3: [imeta.BoundingBox('cup', 1, 25, 32, 95, 95)],         # Slightly misplaced
                id4: [imeta.BoundingBox('cup', 0.75, 25, 32, 95, 95)]       # Reduced confidence and slightly misplaced
            }
        )
        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)

        self.assertIn('cup', result.classes)
        self.assertEqual(4, len(result.scores_by_class['cup']))
        self.assertEqual(1, result.get_max_score('cup'))

    def test_benchmark_takes_only_best_score(self):
        id1 = oid.ObjectId()
        trial_result = MockTrialResult(
            gt_bboxes={
                id1: [imeta.BoundingBox('cup', 1, 15, 22, 100, 100),
                      imeta.BoundingBox('cup', 1, 115, 122, 50, 50),
                      imeta.BoundingBox('cup', 1, 165, 172, 25, 25)]
            },
            bboxes={
                id1: [imeta.BoundingBox('cup', 1, 15, 22, 175, 175)],
            }
        )
        benchmark = bbox_overlap.BoundingBoxOverlapBenchmark()
        result = benchmark.benchmark_results(trial_result)

        expected_precision = 100 * 100 / (175 * 175)    # Actual area is much smaller than returned area
        expected_recall = 1     # Box covers the entire area
        expected_score = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

        self.assertIn('cup', result.classes)
        self.assertEqual(1, len(result.scores_by_class['cup']))
        self.assertEqual(expected_score, result.get_max_score('cup'))
