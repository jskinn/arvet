# Copyright (c) 2017, John Skinner
import abc
import typing
import bson
from pymodm import MongoModel
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.database.pymodm_abc import ABCModelMeta
from arvet.core.image import Image
from arvet.core.sequence_type import ImageSequenceType


class ImageSource(MongoModel, metaclass=ABCModelMeta):
    """
    An abstract class representing a place to get images from.
    This generalizes datasets and simulators

    The new usage structure for image sources is the with statement, i.e.:
    ```
    with image_source as source_handle:
        for timestamp, image in image_source:

    ```
    This allows us to cleanly startup and shutdown the image source (for simulators).

    TODO: We need more ways to interrogate the image source for information about it.
    """

    @abc.abstractmethod
    def __iter__(self) -> typing.Tuple[float, Image]:
        pass

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the id for this vision system
        :return:
        """
        return self._id

    @property
    @abc.abstractmethod
    def sequence_type(self) -> ImageSequenceType:
        """
        Get the type of image sequence produced by this image source.
        For instance, the source may produce sequential images, or disjoint, random images.
        This may change with the configuration of the image source.
        It is useful for determining which sources can run with which algorithms.
        :return: The image sequence type enum
        :rtype arvet.core.image_sequence.ImageSequenceType:
        """
        pass

    @property
    @abc.abstractmethod
    def is_depth_available(self) -> bool:
        """
        Can this image source produce depth images.
        Some algorithms will require depth images
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def is_labels_available(self) -> bool:
        """
        Do images from this source include object bounding boxes and simple labels in their metadata.
        :return: True iff the image metadata includes bounding boxes
        """
        pass

    @property
    @abc.abstractmethod
    def is_masks_available(self) -> bool:
        """
        Do images from this image source include object labels
        :return: True if this image source can produce object labels for each image
        """
        pass

    @property
    @abc.abstractmethod
    def is_normals_available(self) -> bool:
        """
        Do images from this image source include world normals
        :return: True if images have world normals associated with them 
        """
        pass

    @property
    @abc.abstractmethod
    def is_stereo_available(self) -> bool:
        """
        Can this image source produce stereo images.
        Some algorithms only run with stereo images
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def is_stored_in_database(self) -> bool:
        """
        Do this images from this source come from the database.
        If they are, we can associate results with individual images.
        On the other hand, sources that produce transient images like simulators
        cannot make permanent associations between results and images.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def framerate(self) -> float:
        """
        Get the frame rate of the image source.
        That is, the difference in time between successive images.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def camera_intrinsics(self) -> CameraIntrinsics:
        """
        Get the intrinsics of the camera in this image source.
        This allows systems to use the correct calibration.
        We only guarantee that this works after begin is called
        (in case we need to pre-load meta-information where this is provided)
        :return: A metadata.camera_intrinsics.CameraIntrinsics object.
        """
        pass

    @abc.abstractmethod
    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this image source. Pass these to "get_properties", below.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        :param columns:
        :return:
        """
        pass

    # Optional properties for sources of stereo images
    stereo_offset = None
    right_camera_intrinsics = None
