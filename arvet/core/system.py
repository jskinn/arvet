# Copyright (c) 2017, John Skinner
import abc
import typing
from enum import Enum
import bson
import pymodm
import arvet.database.pymodm_abc as pymodm_abc
from arvet.config.path_manager import PathManager
import arvet.util.transform as tf
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.image_source import ImageSource
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image


class StochasticBehaviour(Enum):
    """
    An enum to describe several different possibilities for random behaviour within the system

    DETERMINISTIC:     The system always gives the same result, there is no stochastic process
    SEEDED:            The system contains pseudo-random processes, but they may be controlled using an initial seed
    NON_DETERMINISTIC: The system contains uncontrolled random processes, which cannot be controlled via seed
    """
    DETERMINISTIC = 0
    SEEDED = 1
    NON_DETERMINISTIC = 2


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
    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics, average_timestep: float) -> None:
        """
        Set the intrinsics used by this image source to process images.
        Many systems take this as configuration.
        :param camera_intrinsics: A camera intrinsics object.
        :param average_timestep: The average time interval between frames, 1 / framerate.
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
    def start_trial(self, sequence_type: ImageSequenceType, seed: int = 0) -> None:
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :param seed: A seed to control the random state of the trial. Should be ignored if the system is not SEEDED.
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

    @abc.abstractmethod
    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this system. Pass these to "get_properties", below.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_properties(self, columns: typing.Iterable[str] = None,
                       settings: typing.Mapping[str, typing.Any] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        These may be overridden by the values stored by the system in a trial result,
        allowing us to have dynamic properties that vary between trials.
        :param columns: The list of columns to get the values of
        :param settings: The settings saved in the trial result
        :return:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def is_deterministic(cls) -> StochasticBehaviour:
        """
        Is the visual system deterministic?
        There are 3 different classes of behavior, distinguised by the :
        - Deterministic systems always give the same results for the same inputs
        - Seeded systems are stochastic, but may be controlled with an initial seed, after which the results are fixed
        - Non-deterministic systems give varying results regardless of seed.

        If this is not DETERMINISTIC, it will have to be tested multiple times either varying the seed or the repeat
        because the performance will be inconsistent between runs.

        :return: The appropriate StochasticBehaviour enum value
        :rtype: StochasticBehaviour
        """
        pass

    @classmethod
    def preload_image_data(cls, image: Image) -> None:
        """
        Read some data from an image object, to force it to load pixel data.
        We do this to pre-load the image data into memory, so that when we actually run the system we don't have
        to wait for the images to load
        Stereo vision systems, or RGB-D vision systems should also read right pixels or depth respectively
        :param image:
        :return:
        """
        _ = image.pixels

    @classmethod
    def get_pretty_name(cls) -> str:
        """
        Get a human-readable name for this metric
        :return:
        """
        return cls.__module__ + '.' + cls.__name__

    @classmethod
    def get_instance(cls) -> 'VisionSystem':
        """
        Get an instance of this vision system, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        all_objects = cls.objects.all()
        if all_objects.count() > 0:
            return all_objects.first()
        obj = cls()
        # obj.save()
        return obj
