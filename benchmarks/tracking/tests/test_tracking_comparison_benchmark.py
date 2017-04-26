import unittest
import core.benchmark
import core.trial_comparison
import slam.tracking_state
import benchmarks.tracking.tracking_comparison_benchmark as track_comp


class MockTrialResult:

    def __init__(self, tracking_states):
        self._tracking_states = tracking_states

    @property
    def identifier(self):
        return 'ThisIsAMockTrialResult'

    @property
    def tracking_states(self):
        return self._tracking_states

    @tracking_states.setter
    def tracking_states(self, tracking_states):
        self._tracking_states = tracking_states

    def get_tracking_states(self):
        return self.tracking_states


class TestTrackingComparisonBenchmark(unittest.TestCase):

    def test_benchmark_results_returns_a_benchmark_result(self):
        trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.OK
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.NOT_INITIALIZED
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, core.trial_comparison.TrialComparisonResult)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(trial_result.identifier, result.trial_result)
        self.assertEqual(reference_trial_result.identifier, result.reference_trial_result)

    def test_benchmark_produces_expected_results(self):
        trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.OK,
            2: slam.tracking_state.TrackingState.LOST,
            2.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.6667: slam.tracking_state.TrackingState.OK,
            3: slam.tracking_state.TrackingState.LOST,
            3.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            3.6667: slam.tracking_state.TrackingState.OK,
            4: slam.tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: slam.tracking_state.TrackingState.OK,
            2.6667: slam.tracking_state.TrackingState.OK,
            3: slam.tracking_state.TrackingState.OK,
            3.3333: slam.tracking_state.TrackingState.LOST,
            3.6667: slam.tracking_state.TrackingState.LOST,
            4: slam.tracking_state.TrackingState.LOST
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)

        self.assertEqual(6, len(result.changes))
        self.assertEqual((slam.tracking_state.TrackingState.NOT_INITIALIZED, slam.tracking_state.TrackingState.OK),
                         result.changes[1.6667])
        self.assertEqual((slam.tracking_state.TrackingState.NOT_INITIALIZED, slam.tracking_state.TrackingState.LOST),
                         result.changes[2])
        self.assertEqual((slam.tracking_state.TrackingState.OK, slam.tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[2.3333])
        self.assertEqual((slam.tracking_state.TrackingState.OK, slam.tracking_state.TrackingState.LOST),
                         result.changes[3])
        self.assertEqual((slam.tracking_state.TrackingState.LOST, slam.tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[3.3333])
        self.assertEqual((slam.tracking_state.TrackingState.LOST, slam.tracking_state.TrackingState.OK),
                         result.changes[3.6667])

    def test_benchmark_associates_results(self):
        trial_result = MockTrialResult(tracking_states={
            1.3433: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6767: slam.tracking_state.TrackingState.OK,
            1.99: slam.tracking_state.TrackingState.LOST,
            2.3433: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.6767: slam.tracking_state.TrackingState.OK,
            3.01: slam.tracking_state.TrackingState.LOST,
            3.3233: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            3.6767: slam.tracking_state.TrackingState.OK,
            4.01: slam.tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: slam.tracking_state.TrackingState.OK,
            2.6667: slam.tracking_state.TrackingState.OK,
            3: slam.tracking_state.TrackingState.OK,
            3.3333: slam.tracking_state.TrackingState.LOST,
            3.6667: slam.tracking_state.TrackingState.LOST,
            4: slam.tracking_state.TrackingState.LOST
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)

        self.assertEqual(6, len(result.changes))
        self.assertEqual((slam.tracking_state.TrackingState.NOT_INITIALIZED, slam.tracking_state.TrackingState.OK),
                         result.changes[1.6667])
        self.assertEqual((slam.tracking_state.TrackingState.NOT_INITIALIZED, slam.tracking_state.TrackingState.LOST),
                         result.changes[2])
        self.assertEqual((slam.tracking_state.TrackingState.OK, slam.tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[2.3333])
        self.assertEqual((slam.tracking_state.TrackingState.OK, slam.tracking_state.TrackingState.LOST),
                         result.changes[3])
        self.assertEqual((slam.tracking_state.TrackingState.LOST, slam.tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[3.3333])
        self.assertEqual((slam.tracking_state.TrackingState.LOST, slam.tracking_state.TrackingState.OK),
                         result.changes[3.6667])

    def test_benchmark_fails_for_not_enough_matching_keys(self):
        trial_result = MockTrialResult(tracking_states={
            1.4333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.7667: slam.tracking_state.TrackingState.OK,
            1.9: slam.tracking_state.TrackingState.LOST,
            2.4333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.7667: slam.tracking_state.TrackingState.OK,
            3.1: slam.tracking_state.TrackingState.LOST,
            3.2333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            3.7667: slam.tracking_state.TrackingState.OK,
            4.1: slam.tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: slam.tracking_state.TrackingState.OK,
            2.6667: slam.tracking_state.TrackingState.OK,
            3: slam.tracking_state.TrackingState.OK,
            3.3333: slam.tracking_state.TrackingState.LOST,
            3.6667: slam.tracking_state.TrackingState.LOST,
            4: slam.tracking_state.TrackingState.LOST
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

    def test_offset_adjusts_timestamps(self):
        trial_result = MockTrialResult(tracking_states={
            101.3433: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            101.6767: slam.tracking_state.TrackingState.OK,
            101.99: slam.tracking_state.TrackingState.LOST,
            102.3433: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            102.6767: slam.tracking_state.TrackingState.OK,
            103.01: slam.tracking_state.TrackingState.LOST,
            103.3233: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            103.6767: slam.tracking_state.TrackingState.OK,
            104.01: slam.tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: slam.tracking_state.TrackingState.OK,
            2.6667: slam.tracking_state.TrackingState.OK,
            3: slam.tracking_state.TrackingState.OK,
            3.3333: slam.tracking_state.TrackingState.LOST,
            3.6667: slam.tracking_state.TrackingState.LOST,
            4: slam.tracking_state.TrackingState.LOST
        })

        # Perform the benchmark, this should fail
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

        # Adjust the offset, this should work
        benchmark.offset = 100  # Updates the reference timestamps to match the query ones
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)

        self.assertEqual(6, len(result.changes))
        self.assertEqual((slam.tracking_state.TrackingState.NOT_INITIALIZED, slam.tracking_state.TrackingState.OK),
                         result.changes[1.6667])
        self.assertEqual((slam.tracking_state.TrackingState.NOT_INITIALIZED, slam.tracking_state.TrackingState.LOST),
                         result.changes[2])
        self.assertEqual((slam.tracking_state.TrackingState.OK, slam.tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[2.3333])
        self.assertEqual((slam.tracking_state.TrackingState.OK, slam.tracking_state.TrackingState.LOST),
                         result.changes[3])
        self.assertEqual((slam.tracking_state.TrackingState.LOST, slam.tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[3.3333])
        self.assertEqual((slam.tracking_state.TrackingState.LOST, slam.tracking_state.TrackingState.OK),
                         result.changes[3.6667])

    def test_max_difference_affects_associations(self):
        trial_result = MockTrialResult(tracking_states={
            1.4333: slam.tracking_state.TrackingState.NOT_INITIALIZED,
            11.7667: slam.tracking_state.TrackingState.OK,
            21.9: slam.tracking_state.TrackingState.LOST,
        })
        reference_trial_result = MockTrialResult(tracking_states={
            2.3333: slam.tracking_state.TrackingState.LOST,
            10.6667: slam.tracking_state.TrackingState.LOST,
            20: slam.tracking_state.TrackingState.OK
        })

        # Perform the benchmark, this should fail since the keys are far appart
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, core.benchmark.FailedBenchmark)

        # Adjust the max difference, this should now allow associations between
        benchmark.max_difference = 5
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, core.benchmark.FailedBenchmark)
