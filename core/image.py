# Copyright (c) 2017, John Skinner
import util.dict_utils as du
import metadata.image_metadata as imeta


class Image:

    def __init__(self, data, metadata, additional_metadata=None,
                 depth_data=None, ground_truth_depth_data=None,
                 labels_data=None, world_normals_data=None, **kwargs):
        # noinspection PyArgumentList
        super().__init__(**kwargs)  # kwargs here passes arguments to other constructors for MI
        self._data = data
        self._metadata = (metadata if isinstance(metadata, imeta.ImageMetadata)
                          else imeta.ImageMetadata.deserialize(metadata))
        self._depth_data = depth_data
        self._gt_depth_data = ground_truth_depth_data
        self._labels_data = labels_data
        self._world_normals_data = world_normals_data
        if additional_metadata is None:
            self._additional_metadata = {}
        else:
            self._additional_metadata = additional_metadata

    @property
    def data(self):
        """
        Get the data for this image, as a numpy array.
        :return: A numpy array containing the image data.
        """
        return self._data

    @property
    def camera_location(self):
        """
        The ground-truth location of the viewpoint from which this image was taken.
        This should be expressed as a 3-element numpy array
        :return: The vector location of the viewpoint when the image is taken, in world coordinates
        """
        return self.camera_pose.location if self.camera_pose is not None else None

    @property
    def camera_orientation(self):
        """
        The ground-truth orientation of the image.
        The orientation is a 4-element numpy array, ordered X, Y, Z, W
        :return:
        """
        return self.camera_pose.rotation_quat(False) if self.camera_pose is not None else None

    @property
    def camera_transform_matrix(self):
        """
        Get the 4x4 transform of the camera when the image was taken.
        :return: A 4x4 numpy array describing the camera pose
        """
        return self.camera_pose.transform_matrix if self.camera_pose is not None else None

    @property
    def camera_pose(self):
        """
        Get the underlying Transform object representing the pose of the camera.
        This is useful to do things like convert points or poses to camera-relative
        :return: A Transform object
        """
        return self.metadata.camera_pose

    @property
    def hash(self):
        """
        Shortcut access to the hash of this image data
        :return: The 64-bit xxhash of this image data, for simple equality checking
        """
        return self.metadata.hash

    @property
    def metadata(self):
        """
        An ImageMetadata object containing the metadata we're explicitly interested in for every image
        :return: The ImageMetadata associated with this object
        :rtype: metadata.image_metadata.ImageMetadata
        """
        return self._metadata

    @property
    def additional_metadata(self):
        """
        Get the additional metadata associated with this image.
        This is where the information about how the image was generated goes.
        These keys are not analysed, stored only for archival and reproduction purposes.
        :return: A dictionary of additional information about this image.
        """
        return self._additional_metadata

    @property
    def depth_data(self):
        """
        Get the scene depth for this image, if available.
        Return None if no depth data is available.
        :return: A numpy array, or None if no depth image is available
        """
        return self._depth_data

    @property
    def ground_truth_depth_data(self):
        """
        The ground-truth depth image, from simulation.
        This is distinct from depth data, which will have noise added for simulated images.
        :return: A numpy array, or None if ground-truth depth data is unavailable
        """
        return self._gt_depth_data

    @property
    def labels_data(self):
        """
        Get the image labels
        :return: A numpy array, or None if no labels are available
        """
        return self._labels_data

    @property
    def world_normals_data(self):
        """
        Get the world normals image
        :return: A numpy array, or None if no world normals are available
        """
        return self._world_normals_data


class StereoImage(Image):
    """
    A stereo image, which has all the properties of two images joined together.
    The base image is the left image, properties for the right images need
    to be accessed specifically, using the properties prefixed with 'right_'
    """

    def __init__(self, left_data, right_data,
                 metadata, additional_metadata=None,
                 left_depth_data=None, left_ground_truth_depth_data=None,
                 left_labels_data=None, left_world_normals_data=None,
                 right_depth_data=None, right_ground_truth_depth_data=None,
                 right_labels_data=None, right_world_normals_data=None, **kwargs):
        # Fiddle the arguments to go to the parents, those not listed here will be passed straight through.
        super().__init__(
            data=left_data,
            metadata=metadata,
            additional_metadata=additional_metadata,
            depth_data=left_depth_data,
            ground_truth_depth_data=left_ground_truth_depth_data,
            labels_data=left_labels_data,
            world_normals_data=left_world_normals_data,
            **kwargs)
        self._right_data = right_data
        self._right_depth_data = right_depth_data
        self._right_gt_depth_data = right_ground_truth_depth_data
        self._right_labels_data = right_labels_data
        self._right_world_normals_data = right_world_normals_data

    @property
    def left_data(self):
        """
        The left image in the stereo pair.
        This is the same as the data property
        :return: The left image data, as a numpy array
        """
        return self.data

    @property
    def right_data(self):
        """
        The right image in the stereo pair.
        :return: The right image data, as a numpy array
        """
        return self._right_data

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

    @property
    def left_camera_transform_matrix(self):
        """
        Get the 4x4 transform of the left camera when the image was taken.
        This is the same as 'camera_transform'
        :return: A 4x4 numpy array describing the camera pose
        """
        return self.camera_transform_matrix

    @property
    def left_camera_pose(self):
        """
        Get the underlying Transform object representing the pose of the left camera.
        This is useful to do things like convert points or poses to camera-relative.
        This is the same as 'camera_pose'
        :return: A Transform object
        """
        return self.camera_pose

    @property
    def right_camera_location(self):
        """
        The location of the right camera in the stereo pair.
        :return: The location of the right camera in the stereo pair, as a 3-element numpy array
        """
        return self.right_camera_pose.location if self.right_camera_pose is not None else None

    @property
    def right_camera_orientation(self):
        """
        The orientation of the right camera in the stereo pair.
        :return: The orientation of the right camera, as a 4-element numpy array unit quaternion, ordered X, Y, Z, W
        """
        return self.right_camera_pose.rotation_quat(False) if self.right_camera_pose is not None else None

    @property
    def right_camera_transform_matrix(self):
        """
        The 4x4 homogenous matrix describing the pose of the right camera
        :return: A 4x4 numpy array
        """
        return self.right_camera_pose.transform_matrix if self.right_camera_pose is not None else None

    @property
    def right_camera_pose(self):
        """
        The underlying transform object describing the pose of the right camera.
        :return: The Transform of the right camera
        """
        return self.metadata.right_camera_pose

    @property
    def left_depth_data(self):
        """
        The left depth image.
        This is the same as depth_data.
        :return: A numpy array, or None if no depth data is available.
        """
        return self.depth_data

    @property
    def right_depth_data(self):
        """
        The right depth image.
        :return: A numpy array, or None if no depth data is available.
        """
        return self._right_depth_data

    @property
    def left_ground_truth_depth_data(self):
        """
        The left ground-truth depth image.
        This is the same as ground_truth_depth_data.
        :return: A numpy array, or None if no depth data is available.
        """
        return self.ground_truth_depth_data

    @property
    def right_ground_truth_depth_data(self):
        """
        The right ground-truth depth image.
        :return: A numpy array, or None if no depth data is available.
        """
        return self._right_gt_depth_data

    @property
    def left_labels_data(self):
        """
        The image labels for the left image.
        This is the same as the labels_data.
        :return: A numpy array, or None if no labels are available.
        """
        return self.labels_data

    @property
    def right_labels_data(self):
        """
        The image labels for the right image.
        :return: A numpy array, or None if no labels are available.
        """
        return self._right_labels_data

    @property
    def left_world_normals_data(self):
        """
        Get the world normals for the left camera viewpoint.
        This is the same as the world_normals_data
        :return: A numpy array, or None if no world normals are available.
        """
        return self.world_normals_data

    @property
    def right_world_normals_data(self):
        """
        Get the world normals filename for the right camera viewpoint.
        :return: A numpy array, or None if no world normals are available.
        """
        return self._right_world_normals_data

    @classmethod
    def make_from_images(cls, left_image, right_image):
        """
        Convert two image objects into a stereo image.
        Relatively self-explanatory, note that it prefers metadata
        from the left image over the right image (right image metadata is lost)
        :param left_image: an Image object
        :param right_image: another Image object
        :return: an instance of StereoImage
        """
        return cls(left_data=left_image.data,
                   right_data=right_image.data,
                   left_depth_data=left_image.depth_data,
                   left_labels_data=left_image.labels_data,
                   left_world_normals_data=left_image.world_normals_data,
                   right_depth_data=right_image.depth_data,
                   right_labels_data=right_image.labels_data,
                   right_world_normals_data=right_image.world_normals_data,
                   metadata=imeta.merge_stereo(left_image.metadata, right_image.metadata),
                   additional_metadata=du.defaults(left_image.additional_metadata, right_image.additional_metadata))
