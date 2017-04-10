import core.benchmark
import core.comparison
import core.visual_slam
import core.tracking_state as ts


class BenchmarkTrackingComparisonResult(core.comparison.ComparisonBenchmarkResult):
    """
    Results of comparison

    That is, what is the average time the algorithm spends reporting a lost tracking state.
    The summary statistic you probably want is the mean, it is the mean duration of lost periods
    over the entire run.
    Other available statistics are the median, std deviation, min, and max
    """

    def __init__(self, benchmark_id, trial_result_id, reference_id, new_lost, new_found, settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, reference_id, True, id_)
        self._new_lost = new_lost
        self._new_found = new_found
        self._settings = settings

    @property
    def num_lost(self):
        return len(self.new_lost)

    @property
    def num_found(self):
        return len(self.new_found)

    @property
    def new_lost(self):
        return self._new_lost

    @property
    def new_found(self):
        return self._new_found

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['new_lost'] = self.new_lost
        output['new_found'] = self.new_found
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'new_lost' in serialized_representation:
            kwargs['new_lost'] = serialized_representation['new_lost']
        if 'new_found' in serialized_representation:
            kwargs['new_found'] = serialized_representation['new_found']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkTrackingComparison(core.comparison.ComparisonBenchmark):
    """
    A tool for benchmarking SLAM based on the mean time the algorithm was lost
    """

    def __init__(self):
        """
        Create a Mean Time Lost benchmark.
        No configuration at this stage, it is just counting tracking state changes.
        """
        pass

    @property
    def identifier(self):
        return 'SLAMTrackingComparison'

    def get_settings(self):
        return {}

    def get_trial_requirements(self):
        return {'success': True, 'tracking_stats': {'$exists': True, '$ne': []}}

    def compare_results(self, trial_result, reference_trial_result, reference_dataset_images):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if not isinstance(trial_result, core.visual_slam.SLAMTrialResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Trial was not a slam trial')
        if not isinstance(reference_trial_result, core.visual_slam.SLAMTrialResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Reference trial was not a slam trial')
        if not len(trial_result.tracking_stats) == len(reference_trial_result.tracking_stats):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Tracking statistics lengths did not match')
        new_lost = []
        new_found = []
        for idx in range(0, len(trial_result.tracking_stats)):
            trial_state = trial_result.tracking_stats[idx]
            reference_state = reference_trial_result.tracking_stats[idx]
            if trial_state == ts.TrackingState.LOST and reference_state == ts.TrackingState.OK:
                new_lost.append(idx)
            elif trial_state == ts.TrackingState.OK and reference_state == ts.TrackingState.LOST:
                new_found.append(idx)
        return BenchmarkTrackingComparisonResult(self.identifier, trial_result.identifier,
                                                 reference_trial_result.identifier, new_lost, new_found, {})
