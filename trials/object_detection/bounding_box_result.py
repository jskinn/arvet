import bson.objectid as oid
import core.trial_result
import metadata.image_metadata as imeta


class BoundingBoxResult(core.trial_result.TrialResult):
    """
    Trial result for a object detector producing bounding boxes.
    Contains the list of bounding boxes with class names and confidence; and ground-truth bounding boxes.
    """

    def __init__(self, system_id, bounding_boxes, ground_truth_bounding_boxes, system_settings, id_=None, **kwargs):
        """
        :param system_id: The id of the system producing the result
        :param bounding_boxes: Map of image id to list of detected bounding boxes
        :param ground_truth_bounding_boxes: Map of image id to list of ground truth bounding boxes
        :param system_settings: The settings of the system producing this result
        :param id_: The id of this object, if it is already in the database
        :param kwargs: Additional arguments, excluding 'success', which will be set to 'True'
        """
        kwargs['success'] = True
        super().__init__(system_id=system_id, system_settings=system_settings, id_=id_, **kwargs)
        self._bounding_boxes = bounding_boxes
        self._ground_truth_bounding_boxes = ground_truth_bounding_boxes

    @property
    def bounding_boxes(self):
        """
        The bounding boxes found by the detector
        Is a dictionary mapping from object id to metadata.image_metadata.BoundingBox objects
        :return:
        """
        return self._bounding_boxes

    @property
    def ground_truth_bounding_boxes(self):
        """
        The ground truth bounding boxes for the object.
        Is a dictionary mapping from object ids to metadata.image_metadata.BoundingBox objects
        :return: A dictionary
        """
        return self._ground_truth_bounding_boxes

    def get_bounding_boxes(self):
        """
        Get the keypoints identified by the detector. This is the same as the property.
        :return: A dictionary of image ids matched to lists of bounding boxes
        """
        return self.bounding_boxes

    def get_ground_truth_bounding_boxes(self):
        """
        Get the ground truth bounding boxes
        :return: A dictionary of image ids matched to lists of bounding boxes
        """
        return self.ground_truth_bounding_boxes

    def serialize(self):
        serialized = super().serialize()
        serialized['bounding_boxes'] = {str(identifier): [bbox.serialize() for bbox in bboxes]
                                        for identifier, bboxes in self.bounding_boxes.items()}
        serialized['gt_bounding_boxes'] = {str(identifier): [bbox.serialize() for bbox in bboxes]
                                           for identifier, bboxes in self.ground_truth_bounding_boxes.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'bounding_boxes' in serialized_representation:
            kwargs['bounding_boxes'] = {
                oid.ObjectId(id_str): {imeta.BoundingBox.deserialize(s_bbox) for s_bbox in s_bboxes}
                for id_str, s_bboxes in serialized_representation['bounding_boxes'].items()}
        if 'gt_bounding_boxes' in serialized_representation:
            kwargs['ground_truth_bounding_boxes'] = {
                oid.ObjectId(id_str): {imeta.BoundingBox.deserialize(s_bbox) for s_bbox in s_bboxes}
                for id_str, s_bboxes in serialized_representation['gt_bounding_boxes'].items()}
        return super().deserialize(serialized_representation, db_client, **kwargs)
