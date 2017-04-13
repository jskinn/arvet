import numpy as np
import pickle
import bson
import core.benchmark


class BenchmarkATEResult(core.benchmark.BenchmarkResult):
    """
    Absolute Trajectory Error results.

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id, trial_result_id, translational_error, ate_settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, True, id_, **kwargs)
        self._translational_error = translational_error
        self._ate_settings = ate_settings

    @property
    def num_pairs(self):
        return len(self._translational_error)

    @property
    def rmse(self):
        return np.sqrt(np.dot(self._translational_error, self._translational_error) / self._num_pairs)

    @property
    def mean(self):
        return np.mean(self._translational_error)

    @property
    def median(self):
        return np.median(self._translational_error)

    @property
    def std(self):
        return np.std(self._translational_error)

    @property
    def min(self):
        return np.min(self._translational_error)

    @property
    def max(self):
        return np.max(self._translational_error)

    @property
    def settings(self):
        return self._ate_settings

    def serialize(self):
        output = super().serialize()
        output['trans_error'] = bson.Binary(pickle.dumps(self._translational_error, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'trans_error' in serialized_representation:
            kwargs['translational_error'] = pickle.loads(serialized_representation['trans_error'])
        else:
            kwargs['translational_error'] = np.array(())
        if 'settings' in serialized_representation:
            kwargs['ate_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)
