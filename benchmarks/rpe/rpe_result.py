#Copyright (c) 2017, John Skinner
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
        trans_error = np.array(list(self.translational_error.values()))
        return np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))

    @property
    def trans_mean(self):
        trans_error = np.array(list(self.translational_error.values()))
        return np.mean(trans_error)

    @property
    def trans_median(self):
        trans_error = np.array(list(self.translational_error.values()))
        return np.median(trans_error)

    @property
    def trans_std(self):
        trans_error = np.array(list(self.translational_error.values()))
        return np.std(trans_error)

    @property
    def trans_min(self):
        trans_error = np.array(list(self.translational_error.values()))
        return np.min(trans_error)

    @property
    def trans_max(self):
        trans_error = np.array(list(self.translational_error.values()))
        return np.max(trans_error)

    @property
    def translational_error(self):
        return self._trans_error

    @property
    def rot_rmse(self):
        rot_error = np.array(list(self.rotational_error.values()))
        return np.sqrt(np.dot(rot_error, rot_error)) / len(rot_error)

    @property
    def rot_mean(self):
        rot_error = np.array(list(self.rotational_error.values()))
        return np.mean(rot_error)

    @property
    def rot_median(self):
        rot_error = np.array(list(self.rotational_error.values()))
        return np.median(rot_error)

    @property
    def rot_std(self):
        rot_error = np.array(list(self.rotational_error.values()))
        return np.std(rot_error)

    @property
    def rot_min(self):
        rot_error = np.array(list(self.rotational_error.values()))
        return np.min(rot_error)

    @property
    def rot_max(self):
        rot_error = np.array(list(self.rotational_error.values()))
        return np.max(rot_error)

    @property
    def rotational_error(self):
        return self._rot_error

    @property
    def settings(self):
        return self._rpe_settings

    def serialize(self):
        output = super().serialize()
        output['trans_error'] = bson.Binary(pickle.dumps(self.translational_error,
                                                                 protocol=pickle.HIGHEST_PROTOCOL))
        output['rot_error'] = bson.Binary(pickle.dumps(self.rotational_error, protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trans_error' in serialized_representation:
            kwargs['translational_error'] = pickle.loads(serialized_representation['trans_error'])
        if 'rot_error' in serialized_representation:
            kwargs['rotational_error'] = pickle.loads(serialized_representation['rot_error'])
        if 'settings' in serialized_representation:
            kwargs['rpe_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
