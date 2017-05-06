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
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)
        self._translational_error = translational_error
        self._ate_settings = ate_settings

        # Compute and cache error statistics
        # We expect translational error to be a map of timestamps to error values as single floats.
        error_array = np.array(list(translational_error.values()))
        self._rmse = np.sqrt(np.dot(error_array, error_array) / len(translational_error))
        self._mean = np.mean(error_array)
        self._median = np.median(error_array)
        self._std = np.std(error_array)
        self._min = np.min(error_array)
        self._max = np.max(error_array)

    @property
    def num_pairs(self):
        return len(self.translational_error)

    @property
    def rmse(self):
        return self._rmse

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
    def translational_error(self):
        return self._translational_error

    @property
    def settings(self):
        return self._ate_settings

    def serialize(self):
        output = super().serialize()
        output['trans_error'] = bson.Binary(pickle.dumps(self.translational_error, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trans_error' in serialized_representation:
            kwargs['translational_error'] = pickle.loads(serialized_representation['trans_error'])
        if 'settings' in serialized_representation:
            kwargs['ate_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
