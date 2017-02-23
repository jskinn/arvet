import numpy
import os


class Image:

    def __init__(self, timestamp, filename, camera_location, camera_orientation, additional_metadata=None,
                 depth_filename=None, labels_filename=None, world_normals_filename=None, **kwargs):
        super().__init__(**kwargs)  # The error here is false, this passes
        self._timestamp = timestamp
        self._filename = filename
        self._camera_location = camera_location
        self._camera_orientation = camera_orientation
        self._depth_filename = depth_filename
        self._labels_filename = labels_filename
        self._world_normals_filename = world_normals_filename
        if additional_metadata is None:
            self._additional_metadata = {}
        else:
            self._additional_metadata = additional_metadata

    @property
    def filename(self):
        """
        Get the filename for this image.
        :return:
        """
        return self._filename

    @property
    def timestamp(self):
        """
        Get the timestamp for when this image was taken
        :return:
        """
        return self._timestamp

    @property
    def camera_location(self):
        """
        The ground-truth location of the viewpoint from which this image was taken.
        This should be expressed as a 3-element numpy array
        :return: The vector location of the viewpoint when the image is taken, in world coordinates
        """
        return self._camera_location

    @property
    def camera_orientation(self):
        """
        The ground-truth orientation of the image.
        The orientation is a 4-element numpy array, ordered X, Y, Z, W
        :return:
        """
        return self._camera_orientation

    @property
    def camera_pose(self):
        """
        Get the 4x4 transform of the camera when the image was taken.
        #TODO: Find or make a toolkit for working with vectors and transformation matrices
        :return: A 4x4 numpy array
        """
        q = self.camera_orientation
        nq = numpy.dot(q, q)
        if nq < numpy.finfo(float).eps:
            return numpy.array((
                (1.0, 0.0, 0.0, self.camera_location[0]),
                (0.0, 1.0, 0.0, self.camera_location[1]),
                (0.0, 0.0, 1.0, self.camera_location[2]),
                (0.0, 0.0, 0.0, 1.0)
            ))
        q *= numpy.sqrt(2.0 / nq)
        q = numpy.outer(q, q)
        return numpy.array((
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], self.camera_location[0]),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], self.camera_location[1]),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], self.camera_location[2]),
            (0.0, 0.0, 0.0, 1.0)
        ))

    @property
    def additional_metadata(self):
        """
        Get the additional metadata associated with this image.
        This is where the information about how the image was generated goes.
        :return: A dictionary of additional information about this image.
        """
        return self._additional_metadata

    @property
    def depth_filename(self):
        """
        Get the filename for the scene depth image for this image.
        Return None if no depth data is available.
        :return: Image path as a string, or None if no depth image is available
        """
        return self._depth_filename

    @property
    def labels_filename(self):
        """
        Get the filename for the image labels
        :return: Image path as a string, or None if no labels are available
        """
        return self._labels_filename

    @property
    def world_normals_filename(self):
        """
        Get the filename for the world normals image
        :return: The image path as a string, or None if no world normals are available
        """
        return self._world_normals_filename

    def delete(self):
        """
        Remove the image and associated files, for cleanup.
        This may throw exceptions for any of the files, if they are in use.
        :return: void
        """
        if self.depth_filename is not None:
            if os.path.isfile(self.depth_filename):
                os.remove(self.depth_filename)
            self._depth_filename = None
        if self.labels_filename is not None:
            if os.path.isfile(self.labels_filename):
                os.remove(self.labels_filename)
            self._labels_filename = None
        if self.world_normals_filename is not None:
            if os.path.isfile(self.world_normals_filename):
                os.remove(self.world_normals_filename)
            self._world_normals_filename = None
        if os.path.isfile(self.filename):
            os.remove(self.filename)


class StereoImage(Image):
    """
    A stereo image, which has all the properties of two images joined together.
    The base image is the left image, properties for the right images need
    to be accessed specifically, using the properties prefixed with 'right_'
    """

    def __init__(self, timestamp, left_filename, right_filename,
                 left_camera_location, left_camera_orientation, right_camera_location, right_camera_orientation,
                 additional_metadata=None,
                 left_depth_filename=None, left_labels_filename=None, left_world_normals_filename=None,
                 right_depth_filename=None, right_labels_filename=None, right_world_normals_filename=None, **kwargs):
        # Fiddle the arguments to go to the parents, those not listed here will be passed straight through.
        kwargs['timestamp'] = timestamp,
        kwargs['filename'] = left_filename,
        kwargs['camera_location'] = left_camera_location,
        kwargs['camera_orientation'] = left_camera_orientation,
        kwargs['additional_metadata'] = additional_metadata,
        kwargs['depth_filename'] = left_depth_filename,
        kwargs['labels_filename'] = left_labels_filename,
        kwargs['world_normals_filename'] = left_world_normals_filename
        super().__init__(**kwargs)
        self._right_filename = right_filename
        self._right_camera_location = right_camera_location
        self._right_camera_orientation = right_camera_orientation
        self._right_depth_filename = right_depth_filename
        self._right_labels_filename = right_labels_filename
        self._right_world_normals_filename = right_world_normals_filename

    @property
    def left_filename(self):
        """
        The filename of the left image in the stereo pair.
        This is the same as the filename property
        :return: This image filename as a string
        """
        return self.filename

    @property
    def right_filename(self):
        """
        The filename of hte right image in the stereo pair.
        :return: The image filename as a string
        """
        return self._right_filename

    @property
    def left_camera_location(self):
        """
        The location of the left camera in the stereo pair.
        This is the same as camera_location
        :return: The location of the left camera as a 3 element numpy array
        """
        return self.camera_location

    @property
    def left_camera_orientation(self):
        """
        The orientation of the left camera in the stereo pair.
        This is the same as camera_orientation
        :return: The orientation of the left camera, as a 4 element numpy array unit quaternion, ordered X, Y, Z, W
        """
        return self.camera_orientation

    # TODO: Camera pose as a transform for the left and right image

    @property
    def right_camera_location(self):
        """
        The location of the right camera in the stereo pair.
        :return: The location of the right camera in the stereo pair, as a 3-element numpy array
        """
        return self._right_camera_location

    @property
    def right_camera_orientation(self):
        """
        The orientation of the right camera in the stereo pair.
        :return: The orientation of the right camera, as a 4-element numpy array unit quaternion, ordered X, Y, Z, W
        """
        return self._right_camera_orientation

    @property
    def left_depth_filename(self):
        """
        The filename of the left depth image.
        This is the same as depth_filename.
        :return: The filename as a string, or None if no depth data is available.
        """
        return self.depth_filename

    @property
    def right_depth_filename(self):
        """
        The filename of the right depth image.
        :return: The filename as a string, or None if no depth data is available.
        """
        return self._right_depth_filename

    @property
    def left_labels_filename(self):
        """
        Get the filename for the image labels on the left image.
        This is the same as the labels_filename.
        :return: Image path as a string, or None if no labels are available.
        """
        return self.labels_filename

    @property
    def right_labels_filename(self):
        """
        Get the filename for the image labels on the right image.
        :return: Image path as a string, or None if no labels are available.
        """
        return self._right_labels_filename

    @property
    def left_world_normals_filename(self):
        """
        Get the world normals for the left camera viewpoint.
        This is the same as the world_normals_filename
        :return: The image path as a string, or None if no world normals are available.
        """
        return self.world_normals_filename

    @property
    def right_world_normals_filename(self):
        """
        Get the world normals filename for the right camera viewpoint.
        :return: The filename as a string, or None if no world normals are available.
        """
        return self._right_world_normals_filename

    def delete(self):
        """
        Remove the image and associated files, for cleanup.
        This may throw exceptions for any of the files, if they are in use.
        :return: void
        """
        if self.right_depth_filename is not None:
            if os.path.isfile(self.right_depth_filename):
                os.remove(self.right_depth_filename)
            self._right_depth_filename = None
        if self.right_labels_filename is not None:
            if os.path.isfile(self.right_labels_filename):
                os.remove(self.right_labels_filename)
            self._right_labels_filename = None
        if self.right_world_normals_filename is not None:
            if os.path.isfile(self.right_world_normals_filename):
                os.remove(self.right_world_normals_filename)
            self._right_world_normals_filename = None
        if os.path.isfile(self.right_filename):
            os.remove(self.right_filename)

    @classmethod
    def make_from_images(cls, left_image, right_image):
        return cls(timestamp=left_image.timestamp,
                   left_filename=left_image.filename,
                   right_filename=right_image.filename,
                   left_camera_location=left_image.camera_location,
                   left_camera_orientation=left_image.camera_orientation,
                   right_camera_location=right_image.camera_location,
                   right_camera_orientation=right_image.camera_orientation,
                   left_depth_filename=left_image.depth_filename,
                   left_labels_filename=left_image.labels_filename,
                   left_world_normals_filename=left_image.world_normals_filename,
                   right_depth_filename=right_image.depth_filename,
                   right_labels_filename=right_image.labels_filename,
                   right_world_normals_filename=right_image.world_normals_filename)
