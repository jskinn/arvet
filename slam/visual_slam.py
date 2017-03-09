import core.trial_result
from core.trajectory import Trajectory, serialize_trajectory, deserialize_trajectory
from slam.tracking_state import tracking_state_from_string


class SLAMTrialResult(core.trial_result.TrialResult):
    """
    The results of running a Monocular SLAM system
    """
    def __init__(self, image_source_id, system_id, success, system_settings, id_=None,
                 trajectory=None, tracking_stats=None, dataset_repeats=0, **kwargs):
        super().__init__(image_source_id, system_id, success, system_settings, id_=id_)
        if isinstance(dataset_repeats, int):
            self._dataset_repeats = dataset_repeats
        else:
            self._dataset_repeats = 0
        if isinstance(trajectory, Trajectory):
            self._trajectory = trajectory
        else:
            self._trajectory = Trajectory([])
        if isinstance(tracking_stats, list):
            self._tracking_stats = tracking_stats
        else:
            self._tracking_stats = []

    @property
    def dataset_repeats(self):
        return self._dataset_repeats

    @dataset_repeats.setter
    def dataset_repeats(self, repeats):
        if isinstance(repeats, int):
            self._dataset_repeats = repeats

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory):
        if isinstance(trajectory, Trajectory):
            self._trajectory = trajectory

    @property
    def tracking_stats(self):
        return self._tracking_stats

    @tracking_stats.setter
    def tracking_stats(self, tracking_stats):
        if isinstance(tracking_stats, list):
            self._tracking_stats = tracking_stats

    def serialize(self):
        serialized = super().serialize()
        serialized['trajectory'] = serialize_trajectory(self.trajectory)
        serialized['tracking_stats'] = [str(tracking_state) for tracking_state in self.tracking_stats]
        serialized['dataset_repeats'] = self._dataset_repeats
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'trajectory' in serialized_representation:
            kwargs['trajectory'] = deserialize_trajectory(serialized_representation['trajectory'])
        if 'tracking_stats' in serialized_representation:
            kwargs['tracking_stats'] = [tracking_state_from_string(s_state)
                                        for s_state in serialized_representation['tracking_stats']]
        if 'dataset_repeats' in serialized_representation:
            kwargs['dataset_repeats'] = serialized_representation['dataset_repeats']
        return super().deserialize(serialized_representation, **kwargs)
