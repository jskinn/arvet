# Copyright (c) 2017, John Skinner
import abc
import bson
import pymodm
import arvet.database.pymodm_abc as pymodm_abc
from arvet.config.path_manager import PathManager
import arvet.util.transform as tf
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.image_source import ImageSource
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image


class VisionSystem(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    A Vision system, something that will be run, benchmarked, and analysed by this program.
    This is the standard interface that everything must implement to work with this system.
    All systems must be entities and stored in the database, so that the framework can load them, and 
    """

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the id of this vision system
        :return:
        """
        return self._id

    @property
    @abc.abstractmethod
    def is_deterministic(self) -> bool:
        """
        Is the visual system deterministic.

        If this is false, it will have to be tested multiple times, because the performance will be inconsistent
        between runs.

        :return: True iff the algorithm will produce the same results each time.
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        """
        Is the dataset appropriate for testing this vision system.
        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics) -> None:
        """
        Set the intrinsics used by this image source to process images.
        Many systems take this as configuration.
        :param camera_intrinsics: A camera intrinsics object.
        :return:
        """
        pass

    def set_stereo_offset(self, offset: tf.Transform) -> None:
        """
        Set the stereo baseline for stereo systems.
        Other systems don't need to override this, it will do nothing.
        :param offset: The distance between the stereo cameras, as a float
        :return:
        """
        pass

    def resolve_paths(self, path_manager: PathManager) -> None:
        """
        If the system requires some external data,
        resolve paths to those files using the path manager
        :param path_manager: the path manager, for finding files
        :return: void
        """
        pass

    @abc.abstractmethod
    def start_trial(self, sequence_type: ImageSequenceType) -> None:
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :return: void
        """
        pass

    @abc.abstractmethod
    def process_image(self, image: Image, timestamp: float) -> None:
        """
        Process an image as part of the current run.
        Should automatically start a new trial if none is currently started.
        :param image: The image object for this frame
        :param timestamp: A timestamp or index associated with this image. Sometimes None.
        :return: void
        """
        pass

    @abc.abstractmethod
    def finish_trial(self) -> 'TrialResult':
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        pass

    @classmethod
    def get_pretty_name(cls) -> str:
        """
        Get a human-readable name for this metric
        :return:
        """
        return cls.__module__ + '.' + cls.__name__
