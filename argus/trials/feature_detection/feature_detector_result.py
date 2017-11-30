# Copyright (c) 2017, John Skinner
import cv2
import logging
import pickle
import bson.objectid as oid
import argus.core.trial_result
import argus.util.transform as tf


class FeatureDetectorResult(argus.core.trial_result.TrialResult):
    """
    Trial result for any feature detector.
    Contains the list of key points produced by that detector
    """

    def __init__(self, system_id, keypoints, timestamps, camera_poses, sequence_type, system_settings,
                 keypoints_id=None, id_=None, **kwargs):
        """
        :param system_id: The identifier of the system producing this result
        :param keypoints: A dictionary of image ids to detected keypoints
        :param timestamps: A map of image timestamps to image identifiers
        :param sequence_type: The type of image sequence used to produce this result
        :param system_settings: The settings of the system producing these results
        :param id_: The database id of the object, if it exists
        :param kwargs: Additional keyword arguments
        """
        kwargs['success'] = True
        super().__init__(system_id=system_id, sequence_type=sequence_type, system_settings=system_settings,
                         id_=id_, **kwargs)
        self._keypoints = keypoints
        self._keypoints_id = keypoints_id
        self._timestamps = timestamps
        self._camera_poses = camera_poses

    @property
    def keypoints(self):
        """
        The keypoints identified by the detector.
        This is a map from image id to lists of key points for that image.
        :return: dict
        """
        return self._keypoints

    @property
    def timestamps(self):
        """
        A map of processing timestamps or indexes to image ids,
        as was provided to the system when it was running.
        :return:
        """
        return self._timestamps

    @property
    def camera_poses(self):
        """
        The ground-truth camera pose for each image
        :return:
        """
        return self._camera_poses

    def get_keypoints(self):
        """
        Get the keypoints identified by the detector. This is the same as the property.
        :return: A dictionary of keypoints matched to image timestamp.
        """
        return self.keypoints

    def get_features_by_timestamp(self):
        """
        Get the detected feature points by timestamp rather than image id.
        :return: A dict mapping timestamps to detected feature results
        """
        return {time: self.keypoints[id_] for time, id_ in self.timestamps.items()}

    def save_data(self, db_client):
        """
        Save the keypoints, many keypoints for many images are too big to store in a single document
        :param db_client:
        :return:
        """
        if self._keypoints_id is None:
            s_keypoints = {identifier: [serialize_keypoint(keypoint) for keypoint in keypoints]
                           for identifier, keypoints in self.keypoints.items()}
            self._keypoints_id = db_client.grid_fs.put(pickle.dumps(s_keypoints, protocol=pickle.HIGHEST_PROTOCOL))

    def serialize(self):
        serialized = super().serialize()
        if self._keypoints_id is None:
            logging.getLogger(__name__).warning("Keypoints have not yet been saved,"
                                                " you must call 'save_data' before serialization")
        serialized['keypoints_id'] = self._keypoints_id
        serialized['timestamps'] = [(stamp, str(identifier)) for stamp, identifier in self.timestamps.items()]
        serialized['camera_poses'] = [(stamp, pose.serialize()) for stamp, pose in self.camera_poses.items()]
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'keypoints_id' in serialized_representation:
            kwargs['keypoints_id'] = serialized_representation['keypoints_id']
            if kwargs['keypoints_id'] is not None:
                s_keypoints = pickle.loads(db_client.grid_fs.get(kwargs['keypoints_id']).read())
                kwargs['keypoints'] = {image_id: [deserialize_keypoint(s_keypoint) for s_keypoint in inner_list]
                                       for image_id, inner_list in s_keypoints.items()}
        if 'timestamps' in serialized_representation:
            kwargs['timestamps'] = {stamp: oid.ObjectId(identifier)
                                    for stamp, identifier in serialized_representation['timestamps']}
        if 'camera_poses' in serialized_representation:
            kwargs['camera_poses'] = {stamp: tf.Transform.deserialize(s_trans) for stamp, s_trans
                                      in serialized_representation['camera_poses']}
        return super().deserialize(serialized_representation, db_client, **kwargs)


def serialize_keypoint(keypoint):
    point = keypoint.pt
    return {
        'x': point[0], 'y': point[1],
        'angle': keypoint.angle,
        'class_id': keypoint.class_id,
        'octave': keypoint.octave,
        'response': keypoint.response,
        'size': keypoint.size
    }


def deserialize_keypoint(s_keypoint):
    return cv2.KeyPoint(x=s_keypoint['x'],
                        y=s_keypoint['y'],
                        _angle=s_keypoint['angle'],
                        _class_id=s_keypoint['class_id'],
                        _octave=s_keypoint['octave'],
                        _response=s_keypoint['response'],
                        _size=s_keypoint['size'])
