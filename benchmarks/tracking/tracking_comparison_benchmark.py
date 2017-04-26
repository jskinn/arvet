import util.associate
import core.trial_comparison
import core.benchmark
import benchmarks.tracking.tracking_comparison_result as track_comp_res


class TrackingComparisonBenchmark(core.trial_comparison.TrialComparison):
    """
    A tool for comparing two slam-type algorithms on their tracking state.
    Identifies places where one was lost and the other was not
    """

    def __init__(self, offset=0, max_difference=0.02):
        self._offset = offset
        self._max_difference = max_difference

    @property
    def identifier(self):
        return 'TrackingComparison'

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def max_difference(self):
        return self._max_difference

    @max_difference.setter
    def max_difference(self, max_difference):
        if max_difference >= 0:
            self._max_difference = max_difference

    def get_settings(self):
        return {
            'offset': self.offset,
            'max_difference': self.max_difference
        }

    def get_trial_requirements(self):
        return {'success': True, 'tracking_stats': {'$exists': True, '$ne': []}}

    def compare_trial_results(self, trial_result, reference_trial_result):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        comp_tracking_stats = trial_result.get_tracking_states()
        ref_tracking_stats = reference_trial_result.get_tracking_states()

        matches = util.associate.associate(comp_tracking_stats, ref_tracking_stats,
                                           offset=self.offset, max_difference=self.max_difference)

        if len(matches) < 2:
            return core.benchmark.FailedBenchmark(benchmark_id=self.identifier,
                                                  trial_result_id=trial_result.identifier,
                                                  reason="Not enough matches between tracking statistics")

        changes = {ref_idx: (ref_tracking_stats[ref_idx], comp_tracking_stats[comp_idx])
                   for comp_idx, ref_idx in matches
                   if ref_tracking_stats[ref_idx] != comp_tracking_stats[comp_idx]}
        return track_comp_res.TrackingComparisonResult(benchmark_id=self.identifier,
                                                       trial_result_id=trial_result.identifier,
                                                       reference_id=reference_trial_result.identifier,
                                                       changes=changes,
                                                       settings=self.get_settings())
