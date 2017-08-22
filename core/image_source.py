import abc


class ImageSource(metaclass=abc.ABCMeta):
    """
    An abstract class representing a place to get images from.
    This generalizes datasets from previous versions,
    and simulators, the big addition of this iteration.
    TODO: We need more ways to interrogate the image source for information about it.
    """

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
        pass

    @property
    @abc.abstractmethod
    def supports_random_access(self):
        """
        True iff we can randomly access the images in this source by index.
        We expect len(image_source) to return meaningful values, and
        :return:
        """
        pass

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
    def is_per_pixel_labels_available(self):
        """
        Do images from this image source include object labels
        :return: True if this image source can produce object labels for each image
        """
        return False

    @property
    @abc.abstractmethod
    def is_labels_available(self):
        """
        Do images from this source include object bounding boxes and simple labels in their metadata.
        :return: True iff the image metadata includes bounding boxes
        """
        return False

    @property
    @abc.abstractmethod
    def is_normals_available(self):
        """
        Do images from this image source include world normals
        :return: True if images have world normals associated with them 
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
    def is_stored_in_database(self):
        """
        Do this images from this source come from the database.
        If they are, we can associate results with individual images.
        On the other hand, sources that produce transient images like simulators
        cannot make permanent associations between results and images.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_camera_intrinsics(self):
        """
        Get the intrinsics of the camera in this image source.
        This allows systems to use the correct calibration.
        :return: A metadata.camera_intrinsics.CameraIntrinsics object
        """
        pass

    def get_stereo_baseline(self):
        """
        If this image source is producing stereo images, return the stereo baseline.
        Otherwise, return None.
        :return:
        """
        return None

    @abc.abstractmethod
    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff we have successfully started iteration
        """
        return False

    @abc.abstractmethod
    def get(self, index):
        """
        If this image source supports random access, get an image by element.
        The valid indexes should be integers in the range 0 <= index < len(image_source)
        If it does not, always return None.
        Unlike get_next_image, this does not return the index or timestamp, since that has to be provided
        :param index: The index of the image to get
        :return: An image object, or None if the index is out of range.
        """
        pass

    @abc.abstractmethod
    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images.
        The second return value must always be

        :return: An Image object (see core.image) or None, and an index (or None)
        """
        return None, None

    @abc.abstractmethod
    def is_complete(self):
        """
        Have we got all the images from this source?
        Some sources are infinite, some are not,
        and this method lets those that are not end the iteration.
        :return: True if there are more images to get, false otherwise.
        """
        return False
