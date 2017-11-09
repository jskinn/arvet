# Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import pickle
import util.dict_utils as du
import database.tests.test_entity as entity_test
import benchmarks.tracking.tracking_result as track_res
import benchmarks.tracking.tracking_benchmark as track_bench


class TestTrackingBenchmarkResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return track_res.TrackingBenchmarkResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_id': np.random.randint(10, 20),
            'settings': {}
        })
        if 'lost_intervals' not in kwargs:
            kwargs['lost_intervals'] = [track_bench.LostInterval(start_time=np.random.uniform(i, i + 0.49),
                                                                 end_time=np.random.uniform(i + 0.5, i + 1),
                                                                 distance=np.random.uniform(0, 1000),
                                                                 num_frames=np.random.randint(0, 100))
                                        for i in range(50)]
        if 'total_distance' not in kwargs:
            kwargs['total_distance'] = np.sum(np.array([i.distance for i in kwargs['lost_intervals']])) + 10000
        if 'total_time' not in kwargs:
            kwargs['total_time'] = np.sum(np.array([i.duration for i in kwargs['lost_intervals']])) + 10
        if 'total_frames' not in kwargs:
            kwargs['total_frames'] = np.sum(np.array([i.frames for i in kwargs['lost_intervals']])) + 10000
        return track_res.TrackingBenchmarkResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: TrackingBenchmarkResult
        :param benchmark_result2: TrackingBenchmarkResult
        :return:
        """
        if (not isinstance(benchmark_result1, track_res.TrackingBenchmarkResult) or
                not isinstance(benchmark_result2, track_res.TrackingBenchmarkResult)):
            self.fail('object was not a TrackingBenchmarkResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertEqual(len(benchmark_result1.lost_intervals), len(benchmark_result2.lost_intervals))
        for idx in range(len(benchmark_result1.lost_intervals)):
            self.assertEqual(benchmark_result1.lost_intervals[idx].start_time,
                             benchmark_result2.lost_intervals[idx].start_time)
            self.assertEqual(benchmark_result1.lost_intervals[idx].end_time,
                             benchmark_result2.lost_intervals[idx].end_time)
            self.assertEqual(benchmark_result1.lost_intervals[idx].duration,
                             benchmark_result2.lost_intervals[idx].duration)
            self.assertEqual(benchmark_result1.lost_intervals[idx].distance,
                             benchmark_result2.lost_intervals[idx].distance)
            self.assertEqual(benchmark_result1.lost_intervals[idx].frames,
                             benchmark_result2.lost_intervals[idx].frames)
        self.assertEqual(benchmark_result1.fraction_distance_lost, benchmark_result2.fraction_distance_lost)
        self.assertEqual(benchmark_result1.fraction_time_lost, benchmark_result2.fraction_time_lost)
        self.assertEqual(benchmark_result1.fraction_frames_lost, benchmark_result2.fraction_frames_lost)
        self.assertEqual(benchmark_result1.settings, benchmark_result2.settings)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'intervals':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for BSON
        intervals1 = pickle.loads(s_model1['intervals'])
        intervals2 = pickle.loads(s_model2['intervals'])
        self.assertEqual(len(intervals1), len(intervals2))
        for idx in range(len(intervals1)):
            self.assertEqual(intervals1[idx].start_time, intervals2[idx].start_time)
            self.assertEqual(intervals1[idx].end_time, intervals2[idx].end_time)
            self.assertEqual(intervals1[idx].duration, intervals2[idx].duration)
            self.assertEqual(intervals1[idx].distance, intervals2[idx].distance)
            self.assertEqual(intervals1[idx].frames, intervals2[idx].frames)

    def test_times_lost_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(len(subject.lost_intervals), subject.times_lost)

    def test_mean_distance_is_correct(self):
        subject = self.make_instance()
        distances = np.array([interval.distance for interval in subject.lost_intervals])
        self.assertEqual(np.mean(distances), subject.mean_distance)

    def test_median_distance_is_correct(self):
        subject = self.make_instance()
        distances = np.array([interval.distance for interval in subject.lost_intervals])
        self.assertEqual(np.median(distances), subject.median_distance)

    def test_std_distance_is_correct(self):
        subject = self.make_instance()
        distances = np.array([interval.distance for interval in subject.lost_intervals])
        self.assertEqual(np.std(distances), subject.std_distance)

    def test_min_distance_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(min([interval.distance for interval in subject.lost_intervals]), subject.min_distance)

    def test_max_distance_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(max([interval.distance for interval in subject.lost_intervals]), subject.max_distance)

    def test_total_distance_lost_is_correct(self):
        subject = self.make_instance()
        self.assertAlmostEqual(sum([interval.distance for interval in subject.lost_intervals]),
                               subject.total_distance_lost)

    def test_mean_time_is_correct(self):
        subject = self.make_instance()
        times = np.array([interval.duration for interval in subject.lost_intervals])
        self.assertEqual(np.mean(times), subject.mean_time)

    def test_median_time_is_correct(self):
        subject = self.make_instance()
        times = np.array([interval.duration for interval in subject.lost_intervals])
        self.assertEqual(np.median(times), subject.median_time)

    def test_std_time_is_correct(self):
        subject = self.make_instance()
        times = np.array([interval.duration for interval in subject.lost_intervals])
        self.assertEqual(np.std(times), subject.std_time)

    def test_min_time_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(min([interval.duration for interval in subject.lost_intervals]), subject.min_time)

    def test_max_time_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(max([interval.duration for interval in subject.lost_intervals]), subject.max_time)

    def test_total_time_lost_is_correct(self):
        subject = self.make_instance()
        self.assertAlmostEqual(sum([interval.duration for interval in subject.lost_intervals]), subject.total_time_lost)

    def test_mean_frames_lost_is_correct(self):
        subject = self.make_instance()
        frames = np.array([interval.frames for interval in subject.lost_intervals])
        self.assertEqual(np.mean(frames), subject.mean_frames_lost)

    def test_median_frames_is_correct(self):
        subject = self.make_instance()
        frames = np.array([interval.frames for interval in subject.lost_intervals])
        self.assertEqual(np.median(frames), subject.median_frames_lost)

    def test_std_frames_lost_is_correct(self):
        subject = self.make_instance()
        frames = np.array([interval.frames for interval in subject.lost_intervals])
        self.assertEqual(np.std(frames), subject.std_frames_lost)

    def test_min_frames_lost_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(min([interval.frames for interval in subject.lost_intervals]), subject.min_frames_lost)

    def test_max_frames_lost_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(max([interval.frames for interval in subject.lost_intervals]), subject.max_frames_lost)

    def test_total_frames_lost_is_correct(self):
        subject = self.make_instance()
        self.assertEqual(sum([interval.frames for interval in subject.lost_intervals]), subject.total_frames_lost)
