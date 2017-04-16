import numpy as np
import pickle
import bson
import core.benchmark


class BenchmarkRPEResult(core.benchmark.BenchmarkResult):
    """
    Relative Pose Error results.

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id, trial_result_id, translational_error,
                 rotational_error, rpe_settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)

        self._trans_error = translational_error
        self._rot_error = rotational_error
        self._rpe_settings = rpe_settings

    @property
    def trans_rmse(self):
        return np.sqrt(np.dot(self._trans_error, self._trans_error)) / len(self._trans_error)

    @property
    def trans_mean(self):
        return np.mean(self._trans_error)

    @property
    def trans_median(self):
        return np.median(self._trans_error)

    @property
    def trans_std(self):
        return np.std(self._trans_error)

    @property
    def trans_min(self):
        return np.min(self._trans_error)

    @property
    def trans_max(self):
        return np.max(self._trans_error)

    @property
    def translational_error(self):
        return self._trans_error

    @property
    def rot_rmse(self):
        return np.sqrt(np.dot(self._rot_error, self._rot_error)) / len(self._rot_error)

    @property
    def rot_mean(self):
        return np.mean(self._rot_error)

    @property
    def rot_median(self):
        return np.median(self._rot_error)

    @property
    def rot_std(self):
        return np.std(self._rot_error)

    @property
    def rot_min(self):
        return np.min(self._rot_error)

    @property
    def rot_max(self):
        return np.min(self._rot_error)

    @property
    def rotational_error(self):
        return self._rot_error

    @property
    def settings(self):
        return self._rpe_settings

    def serialize(self):
        output = super().serialize()
        output['translational_error'] = bson.Binary(pickle.dumps(self.translational_error,
                                                                 protocol=pickle.HIGHEST_PROTOCOL))
        output['rotational_error'] = bson.Binary(pickle.dumps(self.rotational_error, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'translational_error' in serialized_representation:
            kwargs['translational_error'] = pickle.loads(serialized_representation['translational_error'])
        if 'rotational_error' in serialized_representation:
            kwargs['rotational_error'] = pickle.loads(serialized_representation['rotational_error'])
        if 'settings' in serialized_representation:
            kwargs['rpe_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)
