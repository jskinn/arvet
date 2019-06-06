# Copyright (c) 2017, John Skinner
import pymodm
from arvet.database.image_field import ImageField
import arvet.metadata.image_metadata as imeta


class Image(pymodm.MongoModel):
    pixels = ImageField(required=True)
    metadata = pymodm.fields.EmbeddedDocumentField(imeta.ImageMetadata, required=True)
    additional_metadata = pymodm.fields.DictField()
    depth = ImageField()
    ground_truth_depth = ImageField()
    normals = ImageField()

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
        return self.metadata.img_hash


class StereoImage(Image):
    """
    A stereo image, which has all the properties of two images joined together.
    The base image is the left image, properties for the right images need
    to be accessed specifically, using the properties prefixed with 'right_'
    """
    right_pixels = ImageField(required=True)
    right_metadata = pymodm.fields.EmbeddedDocumentField(imeta.ImageMetadata, required=True)
    right_depth = ImageField()
    right_ground_truth_depth = ImageField()
    right_normals = ImageField()

    # -------- RIGHT --------
    @property
    def right_camera_location(self):
        """
        The ground-truth location of the viewpoint from which this image was taken.
        This should be expressed as a 3-element numpy array
        :return: The vector location of the viewpoint when the image is taken, in world coordinates
        """
        return self.right_camera_pose.location if self.right_camera_pose is not None else None

    @property
    def right_camera_orientation(self):
        """
        The ground-truth orientation of the image.
        The orientation is a 4-element numpy array, ordered X, Y, Z, W
        :return:
        """
        return self.right_camera_pose.rotation_quat(False) if self.right_camera_pose is not None else None

    @property
    def right_camera_transform_matrix(self):
        """
        Get the 4x4 transform of the camera when the image was taken.
        :return: A 4x4 numpy array describing the camera pose
        """
        return self.right_camera_pose.transform_matrix if self.right_camera_pose is not None else None

    @property
    def right_camera_pose(self):
        """
        Get the underlying Transform object representing the pose of the camera.
        This is useful to do things like convert points or poses to camera-relative
        :return: A Transform object
        """
        return self.right_metadata.camera_pose

    @property
    def right_hash(self):
        """
        Shortcut access to the hash of this image data
        :return: The 64-bit xxhash of this image data, for simple equality checking
        """
        return self.right_metadata.img_hash

    # -------- LEFT --------
    @property
    def left_pixels(self):
        return self.pixels

    @property
    def left_metadata(self):
        return self.metadata

    @property
    def left_depth(self):
        """
        The left depth image.
        This is the same as depth_data.
        :return: A numpy array, or None if no depth data is available.
        """
        return self.depth

    @property
    def left_ground_truth_depth(self):
        """
        The left ground-truth depth image.
        This is the same as ground_truth_depth_data.
        :return: A numpy array, or None if no depth data is available.
        """
        return self.ground_truth_depth

    @property
    def left_normals(self):
        """
        Get the world normals for the left camera viewpoint.
        This is the same as the world_normals_data
        :return: A numpy array, or None if no world normals are available.
        """
        return self.normals

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

    @classmethod
    def make_from_images(cls, left_image: Image, right_image: Image) -> 'StereoImage':
        """
        Convert two image objects into a stereo image.
        Relatively self-explanatory, note that it prefers metadata
        from the left image over the right image (right image metadata is lost)
        :param left_image: an Image object
        :param right_image: another Image object
        :return: an instance of StereoImage
        """
        additional_metadata = left_image.additional_metadata.copy()
        additional_metadata.update(right_image.additional_metadata)
        return cls(pixels=left_image.pixels,
                   right_pixels=right_image.pixels,
                   metadata=left_image.metadata,
                   right_metadata=right_image.metadata,
                   additional_metadata=additional_metadata,
                   depth=left_image.depth,
                   right_depth=right_image.depth,
                   normals=left_image.normals,
                   right_normals=right_image.normals)
