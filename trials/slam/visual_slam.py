import util.transform as tf
import core.trial_result
from slam.tracking_state import tracking_state_from_string


class SLAMTrialResult(core.trial_result.TrialResult):
    """
    The results of running a Monocular SLAM system
    """
    def __init__(self, image_source_id, system_id, system_settings,
                 id_=None, trajectory=None, tracking_stats=None, **kwargs):
        kwargs['success'] = True
        super().__init__(image_source_id, system_id, system_settings, id_=id_, **kwargs)
        self._trajectory = trajectory
        self._tracking_stats = tracking_stats

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory):
        if isinstance(trajectory, dict):
            self._trajectory = trajectory

    @property
    def tracking_stats(self):
        return self._tracking_stats

    @tracking_stats.setter
    def tracking_stats(self, tracking_stats):
        if isinstance(tracking_stats, dict):
            self._tracking_stats = tracking_stats

    def serialize(self):
        serialized = super().serialize()
        serialized['trajectory'] = {timestamp: tf.serialize_transform(pose)
                                    for timestamp, pose in self.trajectory.items()}
        serialized['tracking_stats'] = {timestamp: str(tracking_state)
                                        for timestamp, tracking_state in self.tracking_stats.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'trajectory' in serialized_representation:
            kwargs['trajectory'] = {timestamp: tf.deserialize_transform(s_transform)
                                    for timestamp, s_transform in serialized_representation['trajectory']}
        if 'tracking_stats' in serialized_representation:
            kwargs['tracking_stats'] = {timestamp: tracking_state_from_string(s_state)
                                        for timestamp, s_state in serialized_representation['tracking_stats'].items()}
        return super().deserialize(serialized_representation, **kwargs)
