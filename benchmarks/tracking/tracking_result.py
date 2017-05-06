import numpy as np
import pickle
import bson
import core.benchmark


class TrackingBenchmarkResult(core.benchmark.BenchmarkResult):
    """
    Tracking statistics result.
    This encapsulates a bunch of information about the intervals during which the algorithm is lost.
    We have statistics relative to distance, duration, or number of frames;
    for each of these we can calculate mean, median, std, min, max, total, and fraction of the whole trajectory
    """

    def __init__(self, benchmark_id, trial_result_id, lost_intervals, total_distance, total_time, total_frames,
                 settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)
        self._lost_intervals = lost_intervals
        self._total_distance = total_distance
        self._total_time = total_time
        self._total_frames = total_frames
        self._settings = settings

    @property
    def lost_intervals(self):
        return self._lost_intervals

    @property
    def times_lost(self):
        return len(self._lost_intervals)

    @property
    def distances(self):
        return np.array([interval.distance for interval in self.lost_intervals])

    @property
    def mean_distance(self):
        return np.mean(self.distances)

    @property
    def median_distance(self):
        return np.median(self.distances)

    @property
    def std_distance(self):
        return np.std(self.distances)

    @property
    def min_distance(self):
        return np.min(self.distances)

    @property
    def max_distance(self):
        return np.max(self.distances)

    @property
    def total_distance_lost(self):
        return np.sum(self.distances)

    @property
    def fraction_distance_lost(self):
        return self.total_distance_lost / self._total_distance

    @property
    def durations(self):
        return np.array([interval.duration for interval in self.lost_intervals])

    @property
    def mean_time(self):
        return np.mean(self.durations)

    @property
    def median_time(self):
        return np.median(self.durations)

    @property
    def std_time(self):
        return np.std(self.durations)

    @property
    def min_time(self):
        return np.min(self.durations)

    @property
    def max_time(self):
        return np.max(self.durations)

    @property
    def total_time_lost(self):
        return np.sum(self.durations)

    @property
    def fraction_time_lost(self):
        return self.total_time_lost / self._total_time

    @property
    def frames_lost(self):
        return np.array([interval.frames for interval in self.lost_intervals])

    @property
    def mean_frames_lost(self):
        return np.mean(self.frames_lost)

    @property
    def median_frames_lost(self):
        return np.median(self.frames_lost)

    @property
    def std_frames_lost(self):
        return np.std(self.frames_lost)

    @property
    def min_frames_lost(self):
        return np.min(self.frames_lost)

    @property
    def max_frames_lost(self):
        return np.max(self.frames_lost)

    @property
    def total_frames_lost(self):
        return np.sum(self.frames_lost)

    @property
    def fraction_frames_lost(self):
        return self.total_frames_lost / self._total_frames

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['intervals'] = bson.Binary(pickle.dumps(self.lost_intervals, protocol=pickle.HIGHEST_PROTOCOL))
        output['total_distance'] = self._total_distance
        output['total_time'] = self._total_time
        output['total_frames'] = self._total_frames
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'intervals' in serialized_representation:
            kwargs['lost_intervals'] = pickle.loads(serialized_representation['intervals'])
        if 'total_distance' in serialized_representation:
            kwargs['total_distance'] = serialized_representation['total_distance']
        if 'total_time' in serialized_representation:
            kwargs['total_time'] = serialized_representation['total_time']
        if 'total_frames' in serialized_representation:
            kwargs['total_frames'] = serialized_representation['total_frames']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
