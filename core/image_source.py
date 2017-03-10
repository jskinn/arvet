import abc
import core.image_sequence


class ImageSource(metaclass=abc.ABCMeta):
    """
    An abstract class representing a place to get images from.
    This generalizes datasets from previous versions,
    and simulators, the big addition of this iteration.
    TODO: We need more ways to interrogate the image source for information about it.
    """

    @property
    @abc.abstractmethod
    def is_depth_available(self):
        """
        Can this image source produce depth images.
        Some algorithms will require depth images
        :return:
        """
        return False

    @property
    @abc.abstractmethod
    def is_stereo_available(self):
        """
        Can this image source produce stereo images.
        Some algorithms only run with stereo images
        :return:
        """
        return False

    @property
    @abc.abstractmethod
    def sequence_type(self):
        """
        Get the type of image sequence produced by this image source.
        For instance, the source may produce sequential images, or disjoint, random images.
        This may change with the configuration of the image source.
        It is useful for determining which sources can run with which algorithms.
        :return: The image sequence type enum
        :rtype core.image_sequence.ImageSequenceType:
        """
        return core.image_sequence.ImageSequenceType.NON_SEQUENTIAL

    @abc.abstractmethod
    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff
        """
        return False

    @abc.abstractmethod
    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images

        :return: An Image object (see core.image) or None
        """
        return None

    @abc.abstractmethod
    def is_complete(self):
        """
        Have we got all the images from this source?
        Some sources are infinite, some are not,
        and this method lets those that are not end the iteration.
        :return: True if there are more images to get, false otherwise.
        """
        return False
