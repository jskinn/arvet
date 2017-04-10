import numpy as np
import core.benchmark
import core.visual_slam
import core.tracking_state


class BenchmarkMDLResult(core.benchmark.BenchmarkResult):
    """
    Mean Distance Lost results.

    That is, what is the average time the algorithm spends reporting a lost tracking state.
    The summary statistic you probably want is the mean, it is the mean duration of lost periods
    over the entire run.
    Other available statistics are the median, std deviation, min, and max
    """

    def __init__(self, benchmark_id, trial_result_id, lost_distances, settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, True, id_)

        if isinstance(lost_distances, dict):
            self._times_lost = lost_distances['times_lost']
            self._mean = lost_distances['mean']
            self._median = lost_distances['median']
            self._std = lost_distances['std']
            self._min = lost_distances['min']
            self._max = lost_distances['max']
        else:
            self._times_lost = len(lost_distances)
            if self._times_lost > 0:
                self._mean = np.mean(lost_distances)
                self._median = np.median(lost_distances)
                self._std = np.std(lost_distances)
                self._min = np.min(lost_distances)
                self._max = np.max(lost_distances)
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
        kwargs['lost_distances'] = tracking_times

        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']

        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkMDL(core.benchmark.Benchmark):
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
        return 'MeanDistanceLost'

    def get_settings(self):
        return {}

    def get_trial_requirements(self):
        return {'success': True, 'tracking_stats': {'$exists': True, '$ne': []}}

    def benchmark_results(self, dataset_images, trial_result):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if isinstance(trial_result, core.visual_slam.SLAMTrialResult):
            is_lost = False
            lost_distance = 0
            prev_location = np.array([0, 0, 0])
            lost_distances = []

            dataset_length = len(dataset_images)
            if dataset_length * trial_result.dataset_repeats != len(trial_result.tracking_stats):
                return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                      "Tracking results do not match dataset length,"
                                                      "we had {0} tracking results for {1} images."
                                                      .format(len(trial_result.tracking_stats),
                                                              dataset_length * trial_result.dataset_repeats))

            for repeat in range(trial_result.dataset_repeats):
                for idx, image in enumerate(dataset_images):
                    if is_lost:
                        if idx > 0:
                            difference = image.camera_location - prev_location
                            lost_distance += np.sqrt(np.dot(difference, difference))

                        if (trial_result.tracking_stats[repeat * dataset_length + idx] ==
                                core.tracking_state.TrackingState.OK):
                            is_lost = False
                            lost_distances.append(lost_distance)
                        prev_location = image.camera_location
                    elif (trial_result.tracking_stats[repeat * dataset_length + idx] ==
                              core.tracking_state.TrackingState.LOST):
                        is_lost = True
                        if idx > 0:
                            difference = image.camera_location - prev_location
                            lost_distance = np.sqrt(np.dot(difference, difference))
                        prev_location = image.camera_location

            # We're still lost at the end, add the final distance
            if is_lost:
                lost_distances.append(lost_distance)

            return BenchmarkMDLResult(self.identifier, trial_result.identifier,
                                      np.array(lost_distances), self.get_settings())
        else:
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Trial was not a slam trial')
