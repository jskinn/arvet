# Copyright (c) 2017, John Skinner
from operator import itemgetter, attrgetter
import typing
import pymodm

from arvet.util.column_list import ColumnList
from arvet.database.enum_field import EnumField
from arvet.database.transform_field import TransformField
import arvet.metadata.camera_intrinsics as cam_intr
from arvet.metadata.image_metadata import MaskedObject
from arvet.core.image import Image, StereoImage
from arvet.core.sequence_type import ImageSequenceType
import arvet.core.image_source


class ImageCollection(arvet.core.image_source.ImageSource, pymodm.MongoModel):
    """
    A collection of images stored in the database.
    This can be a sequential set of images like a video, or a random sampling of different pictures.
    """
    images = pymodm.fields.ListField(
        pymodm.ReferenceField(Image, required=True, on_delete=pymodm.fields.ReferenceField.CASCADE),
        required=True
    )
    timestamps = pymodm.fields.ListField(pymodm.fields.FloatField(required=True), required=True)
    sequence_type = EnumField(ImageSequenceType, required=True)
    image_group = pymodm.fields.CharField(required=True)

    is_depth_available = pymodm.fields.BooleanField(required=True)
    is_normals_available = pymodm.fields.BooleanField(required=True)
    is_stereo_available = pymodm.fields.BooleanField(required=True)
    is_labels_available = pymodm.fields.BooleanField(required=True)
    is_masks_available = pymodm.fields.BooleanField(required=True)
    is_stored_in_database = True

    camera_intrinsics = pymodm.fields.EmbeddedDocumentField(cam_intr.CameraIntrinsics, required=True)
    stereo_offset = TransformField()
    right_camera_intrinsics = pymodm.fields.EmbeddedDocumentField(cam_intr.CameraIntrinsics)

    # Extra properties for identifying the sequence and the trajectory
    dataset = pymodm.fields.CharField()     # The name of the dataset. Should be unique when combined with sequence
    sequence_name = pymodm.fields.CharField()   # The name of the sequence within the dataset.
    trajectory_id = pymodm.fields.CharField()   # A unique name for the trajectory, so we can associate results by traj

    # List of available columns, and a getter for retrieving the value of each
    columns = ColumnList(
        dataset=attrgetter('dataset'),
        sequence_name=attrgetter('sequence_name'),
        trajectory_id=attrgetter('trajectory_id')
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If the timestamps aren't sorted, re-sort them
        if not all(self.timestamps[idx] >= self.timestamps[idx - 1] for idx in range(1, len(self.timestamps))):
            pairs = sorted(zip(self.timestamps, self.images), key=itemgetter(0))
            self.images = [pair[1] for pair in pairs]
            self.timestamps = [pair[0] for pair in pairs]

        # Infer missing properties from the images
        if len(self.images) > 0:
            if self.image_group is None or len(self.image_group) <= 0:
                self.image_group = self.images[0].image_group
            if self.is_depth_available is None:
                self.is_depth_available = all(image.depth is not None for image in self.images)
            if self.is_normals_available is None:
                self.is_normals_available = all(image.normals is not None for image in self.images)
            if self.is_stereo_available is None:
                self.is_stereo_available = all(isinstance(image, StereoImage) for image in self.images)
            if self.is_labels_available is None:
                self.is_labels_available = any(len(image.metadata.labelled_objects) > 0 for image in self.images)
            if self.is_masks_available is None:
                self.is_masks_available = all(
                    any(isinstance(label, MaskedObject) for label in image.metadata.labelled_objects)
                    for image in self.images)
            if self.camera_intrinsics is None:
                self.camera_intrinsics = self.images[0].metadata.intrinsics
            if isinstance(self.images[0], StereoImage):
                if self.stereo_offset is None:
                    self.stereo_offset = self.images[0].stereo_offset
                if self.right_camera_intrinsics is None:
                    self.right_camera_intrinsics = self.images[0].right_metadata.intrinsics

        # Default value for trajectory id from the dataset and sequence name
        if self.trajectory_id is None and self.dataset is not None and self.sequence_name is not None:
            self.trajectory_id = self.dataset + ":" + self.sequence_name

    def __len__(self):
        """
        The length of the image collection
        :return:
        """
        return len(self.images)

    def __iter__(self):
        """
        Iterator for the image collection.
        Returns a timestamp and image for each iteration
        :return:
        """
        for timestamp, image in sorted(zip(self.timestamps, self.images), key=itemgetter(0)):
            yield timestamp, image

    def __getitem__(self, item):
        """
        Allow index-based access. Why not.
        This is the same as get
        :param item:
        :return:
        """
        return self.timestamps[item], self.images[item]

    @property
    def average_timestep(self) -> float:
        """
        Get the average time interval between frames.
        :return: The total time divided by 1 less than the number of frames (the number of intervals)
        """
        return (max(self.timestamps) - min(self.timestamps)) / (len(self.timestamps) - 1)

    def get_image_group(self) -> str:
        """
        If the image source is stored in the database, get the image group it is stored under.
        This lets us pre-load images from that group
        :return: The image_group property
        """
        return self.image_group

    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this system. Pass these to "get_properties", below.
        :return:
        """
        return set(self.columns.keys())

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        :param columns:
        :return:
        """
        if columns is None:
            columns = self.columns.keys()
        return {
            col_name: self.columns.get_value(self, col_name)
            for col_name in columns
            if col_name in self.columns
        }
