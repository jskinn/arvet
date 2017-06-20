import cv2
import bson.objectid as oid
import core.trial_result


class FeatureDetectorResult(core.trial_result.TrialResult):
    """
    Trial result for any feature detector.
    Contains the list of key points produced by that detector
    """

    def __init__(self, system_id, keypoints, system_settings, id_=None, **kwargs):
        """

        :param image_source_id:
        :param system_id:
        :param keypoints:
        :param system_settings:
        :param id_:
        :param kwargs:
        """
        kwargs['success'] = True
        super().__init__(system_id=system_id, system_settings=system_settings, id_=id_, **kwargs)
        self._keypoints = keypoints

    @property
    def keypoints(self):
        """
        The keypoints identified by the detector.
        :return:
        """
        return self._keypoints

    def get_keypoints(self):
        """
        Get the keypoints identified by the detector. This is the same as the property.
        :return: A dictionary of keypoints matched to image timestamp.
        """
        return self.keypoints

    def serialize(self):
        serialized = super().serialize()
        serialized['keypoints'] = {str(identifier): [serialize_keypoint(keypoint) for keypoint in keypoints]
                                   for identifier, keypoints in self.keypoints.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'keypoints' in serialized_representation:
            kwargs['keypoints'] = {oid.ObjectId(identifier): [deserialize_keypoint(s_keypoint)
                                                              for s_keypoint in s_keypoints]
                                   for identifier, s_keypoints in serialized_representation['keypoints'].items()}
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
