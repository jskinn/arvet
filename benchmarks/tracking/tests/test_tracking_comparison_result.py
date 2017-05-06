import unittest
import numpy as np
import database.tests.test_entity as entity_test
import util.dict_utils as du
import benchmarks.tracking.tracking_comparison_result as track_comp_res
import trials.slam.tracking_state


class TestTrackingComparisonResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return track_comp_res.TrackingComparisonResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_id': np.random.randint(10, 20),
            'reference_id': np.random.randint(20, 30),
            'settings': {}
        })
        if 'changes' not in kwargs:
            tracking_types = [
                trials.slam.tracking_state.TrackingState.NOT_INITIALIZED,
                trials.slam.tracking_state.TrackingState.LOST,
                trials.slam.tracking_state.TrackingState.OK
            ]
            kwargs['changes'] = {}
            for _ in range(100):
                ref_idx = np.random.randint(0, len(tracking_types))
                comp_idx = (ref_idx + np.random.randint(1, len(tracking_types))) % len(tracking_types)
                kwargs['changes'][np.random.uniform(0, 600)] = (tracking_types[ref_idx], tracking_types[comp_idx])
        return track_comp_res.TrackingComparisonResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: TrackingBenchmarkResult
        :param benchmark_result2: TrackingBenchmarkResult
        :return:
        """
        if (not isinstance(benchmark_result1, track_comp_res.TrackingComparisonResult) or
                not isinstance(benchmark_result2, track_comp_res.TrackingComparisonResult)):
            self.fail('object was not a TrackingComparisonResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_result, benchmark_result2.trial_result)
        self.assertEqual(benchmark_result1.changes, benchmark_result2.changes)
        self.assertEqual(benchmark_result1.new_tracking_count, benchmark_result2.new_tracking_count)
        self.assertEqual(benchmark_result1.new_lost_count, benchmark_result2.new_lost_count)
        self.assertEqual(benchmark_result1.settings, benchmark_result2.settings)

    def test_new_tracking_count_correct(self):
        subject = self.make_instance()
        new_tracking = 0
        for ref_state, comp_state in subject.changes.values():
            if (comp_state == trials.slam.tracking_state.TrackingState.OK and
                    ref_state != trials.slam.tracking_state.TrackingState.OK):
                new_tracking += 1
        self.assertEquals(new_tracking, subject.new_tracking_count)

    def test_new_lost_count_correct(self):
        subject = self.make_instance()
        new_lost = 0
        for ref_state, comp_state in subject.changes.values():
            if (comp_state != trials.slam.tracking_state.TrackingState.OK and
                    ref_state == trials.slam.tracking_state.TrackingState.OK):
                new_lost += 1
        self.assertEquals(new_lost, subject.new_lost_count)

    def test_initialized_is_lost_changes_new_tracking_count(self):
        subject = self.make_instance()
        # make sure there is at least one different result
        subject.changes[601.2] = (trials.slam.tracking_state.TrackingState.LOST,
                                  trials.slam.tracking_state.TrackingState.NOT_INITIALIZED)
        initial_count = subject.new_tracking_count

        subject.initializing_is_lost = False    # Default is True
        new_tracking = 0
        for ref_state, comp_state in subject.changes.values():
            if (comp_state != trials.slam.tracking_state.TrackingState.LOST and
                    ref_state == trials.slam.tracking_state.TrackingState.LOST):
                new_tracking += 1
        self.assertNotEqual(initial_count, subject.new_tracking_count)
        self.assertEqual(new_tracking, subject.new_tracking_count)

    def test_initialized_is_lost_changes_new_lost_count(self):
        subject = self.make_instance()
        initial_count = subject.new_lost_count

        subject.initializing_is_lost = False    # Default is True
        new_lost = 0
        for ref_state, comp_state in subject.changes.values():
            if (comp_state == trials.slam.tracking_state.TrackingState.LOST and
                    ref_state != trials.slam.tracking_state.TrackingState.LOST):
                new_lost += 1
        self.assertNotEqual(initial_count, subject.new_lost_count)
        self.assertEqual(new_lost, subject.new_lost_count)
