import bson.objectid as oid
import core.trial_result
import util.dict_utils as du


class BoundingBox:
    """
    A bounding box.
    x and y are column-row of the top left corner, height and width extending down and right from there.
    Origin is top left corner of the image.
    Bounding boxes should be mapped to a particular image,
    and image coordinates are relative to the base resolution of that image.
    """
    def __init__(self, class_names, confidence, x, y, width, height):
        self.class_names = tuple(class_names)
        self.confidence = float(confidence)
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def __eq__(self, other):
        """
        Override equals. Bounding boxes are equal if the have the same class, confidence, and shape
        :param other:
        :return:
        """
        return (hasattr(other, 'class_names') and
                hasattr(other, 'confidence') and
                hasattr(other, 'x') and
                hasattr(other, 'y') and
                hasattr(other, 'height') and
                hasattr(other, 'width') and
                self.class_names == other.class_names and
                self.confidence == other.confidence and
                self.x == other.x and
                self.y == other.y and
                self.height == other.height and
                self.width == other.width)

    def __hash__(self):
        """
        Hash this object, so it can be in sets
        :return:
        """
        return hash((self.class_names, self.confidence, self.x, self.y, self.height, self.width))

    def serialize(self):
        return {
            'class_names': list(self.class_names),
            'confidence': self.confidence,
            'x': self.x,
            'y': self.y,
            'height': self.height,
            'width': self.width
        }

    @classmethod
    def deserialize(cls, serialized):
        kwargs = {}
        if 'class_names' in serialized:
            kwargs['class_names'] = tuple(serialized['class_names'])
        if 'confidence' in serialized:
            kwargs['confidence'] = serialized['confidence']
        if 'x' in serialized:
            kwargs['x'] = serialized['x']
        if 'y' in serialized:
            kwargs['y'] = serialized['y']
        if 'width' in serialized:
            kwargs['width'] = serialized['width']
        if 'height' in serialized:
            kwargs['height'] = serialized['height']
        return cls(**serialized)


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
                oid.ObjectId(id_str): tuple(BoundingBox.deserialize(s_bbox) for s_bbox in s_bboxes)
                for id_str, s_bboxes in serialized_representation['bounding_boxes'].items()}
        if 'gt_bounding_boxes' in serialized_representation:
            kwargs['ground_truth_bounding_boxes'] = {
                oid.ObjectId(id_str): tuple(BoundingBox.deserialize(s_bbox) for s_bbox in s_bboxes)
                for id_str, s_bboxes in serialized_representation['gt_bounding_boxes'].items()}
        return super().deserialize(serialized_representation, db_client, **kwargs)
