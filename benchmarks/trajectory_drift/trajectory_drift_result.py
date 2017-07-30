import numpy as np
import core.benchmark


class TrajectoryDriftBenchmarkResult(core.benchmark.BenchmarkResult):
    """
    Results from measuring the trajectory drift as is done for the KITTI benchmark.
    """

    def __init__(self, benchmark_id, trial_result_id, errors, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_id=trial_result_id, id_=id_, **kwargs)
        self._errors = errors
        self._settings = settings

    @property
    def raw_errors(self):
        return self._errors

    @property
    def trans_rmse(self):
        trans_error = np.array(list(self.translational_error))
        return np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))

    @property
    def trans_mean(self):
        trans_error = np.array(list(self.translational_error))
        return np.mean(trans_error)

    @property
    def trans_median(self):
        trans_error = np.array(list(self.translational_error))
        return np.median(trans_error)

    @property
    def trans_std(self):
        trans_error = np.array(list(self.translational_error))
        return np.std(trans_error)

    @property
    def trans_min(self):
        trans_error = np.array(list(self.translational_error))
        return np.min(trans_error)

    @property
    def trans_max(self):
        trans_error = np.array(list(self.translational_error))
        return np.max(trans_error)

    @property
    def translational_error(self):
        return [err['t_err'] for err in self.raw_errors]

    @property
    def rot_rmse(self):
        rot_error = np.array(list(self.rotational_error))
        return np.sqrt(np.dot(rot_error, rot_error)) / len(rot_error)

    @property
    def rot_mean(self):
        rot_error = np.array(list(self.rotational_error))
        return np.mean(rot_error)

    @property
    def rot_median(self):
        rot_error = np.array(list(self.rotational_error))
        return np.median(rot_error)

    @property
    def rot_std(self):
        rot_error = np.array(list(self.rotational_error))
        return np.std(rot_error)

    @property
    def rot_min(self):
        rot_error = np.array(list(self.rotational_error))
        return np.min(rot_error)

    @property
    def rot_max(self):
        rot_error = np.array(list(self.rotational_error))
        return np.max(rot_error)

    @property
    def rotational_error(self):
        return [err['r_err'] for err in self.raw_errors]

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['errors'] = self.raw_errors
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'errors' in serialized_representation:
            kwargs['errors'] = serialized_representation['errors']
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
