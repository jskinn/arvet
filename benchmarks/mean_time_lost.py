from numpy import array as nparray, mean, median, std, min, max
from core.benchmark import Benchmark, BenchmarkResult, FailedBenchmark
from core.visual_slam import SLAMTrialResult
from core.tracking_state import TrackingState


class BenchmarkMTLResult(BenchmarkResult):
    """
    Mean Time Lost results.

    That is, what is the average time the algorithm spends reporting a lost tracking state.
    The summary statistic you probably want is the mean, it is the mean duration of lost periods
    over the entire run.
    Other available statistics are the median, std deviation, min, and max
    """

    def __init__(self, benchmark_id, trial_result_id, tracking_times, settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, True, id_)

        if isinstance(tracking_times, dict):
            self._times_lost = tracking_times['times_lost']
            self._mean = tracking_times['mean']
            self._median = tracking_times['median']
            self._std = tracking_times['std']
            self._min = tracking_times['min']
            self._max = tracking_times['max']
        else:
            self._times_lost = len(tracking_times)
            if self._times_lost > 0:
                self._mean = mean(tracking_times)
                self._median = median(tracking_times)
                self._std = std(tracking_times)
                self._min = min(tracking_times)
                self._max = max(tracking_times)
            else:
                self._mean = 0
                self._median = 0
                self._std = 0
                self._min = 0
                self._max = 0

        self._settings = settings

    @property
    def times_lost(self):
        return self._times_lost

    @property
    def mean(self):
        return self._mean

    @property
    def median(self):
        return self._median

    @property
    def std(self):
        return self._std

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['times_lost'] = self.times_lost
        output['mean'] = self.mean
        output['median'] = self.median
        output['std'] = self.std
        output['min'] = self.min
        output['max'] = self.max
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        tracking_times = {}
        if 'times_lost' in serialized_representation:
            tracking_times['times_lost'] = serialized_representation['times_lost']
        if 'mean' in serialized_representation:
            tracking_times['mean'] = serialized_representation['mean']
        if 'median' in serialized_representation:
            tracking_times['median'] = serialized_representation['median']
        if 'std' in serialized_representation:
            tracking_times['std'] = serialized_representation['std']
        if 'min' in serialized_representation:
            tracking_times['min'] = serialized_representation['min']
        if 'max' in serialized_representation:
            tracking_times['max'] = serialized_representation['max']
        kwargs['tracking_times'] = tracking_times

        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']

        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkMTL(Benchmark):
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
        return 'MeanTimeLost'

    def get_settings(self):
        return {}

    def get_trial_requirements(self):
        return {'success': True, 'tracking_stats': { '$exists': True, '$ne': []}}

    def benchmark_results(self, dataset_images, trial_result):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if isinstance(trial_result, SLAMTrialResult):
            lost_start_time = 0  # we start lost
            lost_durations = []

            dataset_length = len(dataset_images)
            if dataset_length * trial_result.dataset_repeats != len(trial_result.tracking_stats):
                return FailedBenchmark(self.identifier, trial_result.identifier,
                                       "Tracking results do not match dataset length,"
                                       "we had {0} tracking results for {1} images."
                                       .format(len(trial_result.tracking_stats),
                                               dataset_length * trial_result.dataset_repeats))

            for repeat in range(trial_result.dataset_repeats):
                for idx, image in enumerate(dataset_images):
                    if (trial_result.tracking_stats[repeat * dataset_length + idx] == TrackingState.LOST and
                            lost_start_time < 0):
                        lost_start_time = image.timestamp
                    elif (trial_result.tracking_stats[repeat * dataset_length + idx] == TrackingState.OK and
                            lost_start_time >= 0):
                        lost_durations.append(image.timestamp - lost_start_time)
                        lost_start_time = -1

            # Were still lost at the end, add the final distance
            if lost_start_time >= 0:
                lost_durations.append(dataset_images.dataset.duration - lost_start_time)

            return BenchmarkMTLResult(self.identifier, trial_result.identifier,
                                      nparray(lost_durations), self.get_settings())
        else:
            return FailedBenchmark(self.identifier, trial_result.identifier, 'Trial was not a slam trial')
