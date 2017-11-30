# Copyright (c) 2017, John Skinner
import argus.core.trial_result
import argus.util.transform as tf


class VisualOdometryResult(argus.core.trial_result.TrialResult):
    """
    The results of running visual odometry over a dataset.
    This should be consistent for all VO algorithms,
    is a set of frame deltas, pose changes over each frame,
    along with the ground trug
    """

    def __init__(self, system_id, sequence_type, system_settings, frame_deltas,
                 ground_truth_trajectory, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(system_id=system_id, sequence_type=sequence_type,
                         system_settings=system_settings, id_=id_, **kwargs)
        self._frame_deltas = frame_deltas
        self._ground_truth_trajectory = ground_truth_trajectory

    @property
    def frame_deltas(self):
        """
        Get the relative pose of each frame relative to the previous frame
        Structure is {timestamp: relative_pose}
        :return:
        """
        return self._frame_deltas

    @property
    def ground_truth_trajectory(self):
        """
        The ground-truth trajectory as absolute poses.
        Structure is {timestamp: absolute_pose}
        Note that this is not directly comparable to frame deltas
        :return:
        """
        return self._ground_truth_trajectory

    def get_ground_truth_camera_poses(self):
        """
        Get the ground-truth camera poses, as a map from timestamp to absolute pose
        :return:
        """
        return self._ground_truth_trajectory

    def get_computed_camera_poses(self):
        """
        Get the computed poses, as a map from timestamp to absolute pose
        This assumes that the first frame is the origin, and builds
        the trajectory from there, assuming
        :return:
        """
        pairs = sorted((timestamp, pose) for timestamp, pose in self.frame_deltas.items())
        current_pose = tf.Transform()
        computed_poses = {}
        for timestamp, delta in pairs:
            delta = delta.find_relative(tf.Transform())     # Flip the direction from previous pose relative to new
            current_pose = current_pose.find_independent(delta)
            computed_poses[timestamp] = current_pose
        return computed_poses

    def serialize(self):
        serialized = super().serialize()
        serialized['frame_deltas'] = [(stamp, pose.serialize()) for stamp, pose in self.frame_deltas.items()]
        serialized['ground_truth_trajectory'] = [(stamp, pose.serialize()) for stamp, pose
                                                 in self.ground_truth_trajectory.items()]
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'frame_deltas' in serialized_representation:
            kwargs['frame_deltas'] = {stamp: tf.Transform.deserialize(s_trans) for stamp, s_trans
                                      in serialized_representation['frame_deltas']}
        if 'ground_truth_trajectory' in serialized_representation:
            kwargs['ground_truth_trajectory'] = {stamp: tf.Transform.deserialize(s_transform) for stamp, s_transform
                                                 in serialized_representation['ground_truth_trajectory']}
        return super().deserialize(serialized_representation, db_client, **kwargs)
