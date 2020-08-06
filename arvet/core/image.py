# Copyright (c) 2017, John Skinner
import typing
import numpy as np
from operator import attrgetter
import pymodm
import pymodm.fields as fields

from arvet.util.transform import Transform
from arvet.util.column_list import ColumnList
import arvet.database.image_manager as image_manager
import arvet.metadata.image_metadata as imeta


class Image(pymodm.MongoModel):
    pixel_path = fields.CharField(required=True)
    image_group = fields.CharField(required=True, blank=False)
    metadata = fields.EmbeddedDocumentField(imeta.ImageMetadata, required=True)
    additional_metadata = pymodm.fields.DictField()
    depth_path = fields.CharField()
    true_depth_path = fields.CharField()
    normals_path = fields.CharField()

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        pixel_path=attrgetter('pixel_path'),
        image_group=attrgetter('image_group'),
        source_type=attrgetter('metadata.source_type'),
        lens_focal_distance=attrgetter('metadata.lens_focal_distance'),
        aperture=attrgetter('metadata.aperture'),

        pos_x=lambda obj: obj.camera_location[0] if obj.camera_location is not None else None,
        pos_y=lambda obj: obj.camera_location[1] if obj.camera_location is not None else None,
        pos_z=lambda obj: obj.camera_location[2] if obj.camera_location is not None else None,

        red_mean=attrgetter('metadata.red_mean'),
        red_std=attrgetter('metadata.red_std'),
        green_mean=attrgetter('metadata.green_mean'),
        green_std=attrgetter('metadata.green_std'),
        blue_mean=attrgetter('metadata.blue_mean'),
        blue_std=attrgetter('metadata.blue_std'),
        depth_mean=attrgetter('metadata.depth_mean'),
        depth_std=attrgetter('metadata.depth_std'),

        environment_type=attrgetter('metadata.environment_type'),
        light_level=lambda obj: obj.metadata.light_level.value if obj.metadata.light_level is not None else None,
        time_of_day=attrgetter('metadata.time_of_day'),

        simulation_world=attrgetter('metadata.simulation_world'),
        lighting_model=attrgetter('metadata.lighting_model'),
        texture_mipmap_bias=attrgetter('metadata.texture_mipmap_bias'),
        normal_maps_enabled=attrgetter('metadata.normal_maps_enabled'),
        roughness_enabled=attrgetter('metadata.roughness_enabled'),
        geometry_decimation=attrgetter('metadata.geometry_decimation')
    )

    def __init__(self, *args, **kwargs):
        # Fudge the constructor keyword arguments to pull out images passed to the constructor
        pixels = None
        depth = None
        true_depth = None
        normals = None
        if 'pixels' in kwargs:
            pixels = kwargs['pixels']
            del kwargs['pixels']
        if 'depth' in kwargs:
            depth = kwargs['depth']
            del kwargs['depth']
        if 'true_depth' in kwargs:
            true_depth = kwargs['true_depth']
            del kwargs['true_depth']
        if 'normals' in kwargs:
            normals = kwargs['normals']
            del kwargs['normals']

        # Call the superclass constructor
        super(Image, self).__init__(*args, **kwargs)

        # Local cache the images to avoid loading them from the database each time
        self._pixels = None
        self._depth = None
        self._true_depth = None
        self._normals = None

        # Set image properties passed to the constructor
        if self.image_group is not None and len(self.image_group) > 0:
            if pixels is not None and (self.pixel_path is None or len(self.pixel_path) == 0):
                self.store_pixels(pixels)
            if depth is not None and (self.depth_path is None or len(self.depth_path) == 0):
                self.store_depth(depth)
            if true_depth is not None and (self.true_depth_path is None or len(self.true_depth_path) == 0):
                self.store_true_depth(true_depth)
            if normals is not None and (self.normals_path is None or len(self.normals_path) == 0):
                self.store_normals(normals)

    @property
    def pixels(self) -> typing.Union[None, np.ndarray]:
        if self._pixels is None and self.pixel_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.pixel_path):
                    self._pixels = image_group.get_image(self.pixel_path)
        return self._pixels

    @property
    def depth(self) -> typing.Union[None, np.ndarray]:
        if self._depth is None and self.depth_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.depth_path):
                    self._depth = image_group.get_image(self.depth_path)
        return self._depth

    @property
    def true_depth(self) -> typing.Union[None, np.ndarray]:
        if self._true_depth is None and self.true_depth_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.true_depth_path):
                    self._true_depth = image_group.get_image(self.true_depth_path)
        return self._true_depth

    @property
    def normals(self) -> typing.Union[None, np.ndarray]:
        if self._normals is None and self.normals_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.normals_path):
                    self._normals = image_group.get_image(self.normals_path)
        return self._normals

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
    def camera_pose(self) -> Transform:
        """
        Get the underlying Transform object representing the pose of the camera.
        This is useful to do things like convert points or poses to camera-relative
        :return: A Transform object
        """
        return self.metadata.camera_pose

    @property
    def hash(self) -> bytes:
        """
        Shortcut access to the hash of this image data
        :return: The 64-bit xxhash of this image data, for simple equality checking
        """
        return self.metadata.img_hash

    def store_pixels(self, pixels: np.ndarray):
        """
        Save a set of pixels for this image
        Updates pixel_path
        :param pixels: The actual image pixels for this image
        :return:
        """
        self._pixels = pixels
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.pixel_path is not None and image_group.is_valid_path(self.pixel_path):
                # Remove the old value. This should be rare.
                image_group.remove_image(self.pixel_path)
            self.pixel_path = image_group.store_image(pixels)

    def store_depth(self, depth: np.ndarray):
        """
        Save a depth image for this image
        Updates depth_path
        :param depth: The depth image
        :return:
        """
        self._depth = depth
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.depth_path is not None and image_group.is_valid_path(self.depth_path):
                image_group.remove_image(self.depth_path)
            self.depth_path = image_group.store_image(depth)

    def store_true_depth(self, true_depth: np.ndarray):
        """
        Save a true depth image for this image
        Updates true_depth_path
        :param true_depth: The depth image
        :return:
        """
        self._true_depth = true_depth
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.true_depth_path is not None and image_group.is_valid_path(self.true_depth_path):
                image_group.remove_image(self.true_depth_path)
            self.true_depth_path = image_group.store_image(true_depth)

    def store_normals(self, normals: np.ndarray):
        """
        Save a normals image for this image
        Updates normals_path
        :param normals: The surface normals
        :return:
        """
        self._normals = normals
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.normals_path is not None and image_group.is_valid_path(self.normals_path):
                image_group.remove_image(self.normals_path)
            self.normals_path = image_group.store_image(normals)

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


class StereoImage(Image):
    """
    A stereo image, which has all the properties of two images joined together.
    The base image is the left image, properties for the right images need
    to be accessed specifically, using the properties prefixed with 'right_'
    """
    right_pixel_path = fields.CharField(required=True)
    right_metadata = pymodm.fields.EmbeddedDocumentField(imeta.ImageMetadata, required=True)
    right_depth_path = fields.CharField()
    right_true_depth_path = fields.CharField()
    right_normals_path = fields.CharField()

    # Columns, extending the image columns, above
    columns = ColumnList(
        Image.columns,
        stereo_offset=lambda obj: np.linalg.norm(obj.stereo_offset.location) if obj.stereo_offset is not None else None,
        right_lens_focal_distance=attrgetter('right_metadata.lens_focal_distance'),
        right_aperture=attrgetter('right_metadata.aperture'),

        right_red_mean=attrgetter('right_metadata.red_mean'),
        right_red_std=attrgetter('right_metadata.red_std'),
        right_green_mean=attrgetter('right_metadata.green_mean'),
        right_green_std=attrgetter('right_metadata.green_std'),
        right_blue_mean=attrgetter('right_metadata.blue_mean'),
        right_blue_std=attrgetter('right_metadata.blue_std'),
        right_depth_mean=attrgetter('right_metadata.depth_mean'),
        right_depth_std=attrgetter('right_metadata.depth_std'),
    )

    def __init__(self, *args, **kwargs):
        # Pull out and reinterpret image arguments passed to the constructor
        right_pixels = None
        right_depth = None
        right_true_depth = None
        right_normals = None
        if 'right_pixels' in kwargs:
            right_pixels = kwargs['right_pixels']
            del kwargs['right_pixels']
        if 'right_depth' in kwargs:
            right_depth = kwargs['right_depth']
            del kwargs['right_depth']
        if 'right_true_depth' in kwargs:
            right_true_depth = kwargs['right_true_depth']
            del kwargs['right_true_depth']
        if 'right_normals' in kwargs:
            right_normals = kwargs['right_normals']
            del kwargs['right_normals']

        # Call the superclass constructor
        super(StereoImage, self).__init__(*args, **kwargs)

        # Cache the images to prevent unnecessary loading
        self._right_pixels = None
        self._right_depth = None
        self._right_true_depth = None
        self._right_normals = None

        # Set image properties passed to the constructor
        if self.image_group is not None and len(self.image_group) > 0:
            if right_pixels is not None and (self.right_pixel_path is None or len(self.right_pixel_path) == 0):
                self.store_right_pixels(right_pixels)
            if right_depth is not None and (self.right_depth_path is None or len(self.right_depth_path) == 0):
                self.store_right_depth(right_depth)
            if right_true_depth is not None and (
                    self.right_true_depth_path is None or len(self.right_true_depth_path) == 0):
                self.store_right_true_depth(right_true_depth)
            if right_normals is not None and (self.right_normals_path is None or len(self.right_normals_path) == 0):
                self.store_right_normals(right_normals)

    # -------- RIGHT --------

    @property
    def right_pixels(self) -> typing.Union[None, np.ndarray]:
        if self._right_pixels is None and self.right_pixel_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.right_pixel_path):
                    self._right_pixels = image_group.get_image(self.right_pixel_path)
        return self._right_pixels

    @property
    def right_depth(self) -> typing.Union[None, np.ndarray]:
        if self._right_pixels is None and self.right_depth_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.right_depth_path):
                    self._right_depth = image_group.get_image(self.right_depth_path)
        return self._right_depth

    @property
    def right_true_depth(self) -> typing.Union[None, np.ndarray]:
        if self._right_true_depth is None and self.right_true_depth_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.right_true_depth_path):
                    self._right_true_depth = image_group.get_image(self.right_true_depth_path)
        return self._right_true_depth

    @property
    def right_normals(self) -> typing.Union[None, np.ndarray]:
        if self._right_normals is None and self.right_normals_path is not None:
            with image_manager.get().get_group(self.image_group) as image_group:
                if image_group.is_valid_path(self.right_normals_path):
                    self._right_normals = image_group.get_image(self.right_normals_path)
        return self._right_normals

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
    def right_camera_pose(self) -> Transform:
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

    def store_right_pixels(self, pixels: np.ndarray):
        """
        Save a set of pixels for the right image
        Updates right_pixel_path
        :param pixels: The actual right image pixels for this image
        """
        self._right_pixels = pixels
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.right_pixel_path is not None and image_group.is_valid_path(self.right_pixel_path):
                image_group.remove_image(self.right_pixel_path)
            self.right_pixel_path = image_group.store_image(pixels)

    def store_right_depth(self, depth: np.ndarray):
        """
        Save a right image depth image for this image
        Updates right_depth_path
        :param depth: The depth image
        """
        self._right_depth = depth
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.right_depth_path is not None and image_group.is_valid_path(self.right_depth_path):
                image_group.remove_image(self.right_depth_path)
            self.right_depth_path = image_group.store_image(depth)

    def store_right_true_depth(self, true_depth: np.ndarray):
        """
        Save a true depth right side image for this image
        Updates true_depth_path
        :param true_depth: The depth image
        :return:
        """
        self._right_true_depth = true_depth
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.right_true_depth_path is not None and image_group.is_valid_path(self.right_true_depth_path):
                image_group.remove_image(self.right_true_depth_path)
            self.right_true_depth_path = image_group.store_image(true_depth)

    def store_right_normals(self, normals: np.ndarray):
        """
        Save a right normals image for this image
        Updates normals_path
        :param normals: The surface normals
        :return:
        """
        self._right_normals = normals
        with image_manager.get().get_group(self.image_group, allow_write=True) as image_group:
            if self.right_normals_path is not None and image_group.is_valid_path(self.right_normals_path):
                image_group.remove_image(self.right_normals_path)
            self.right_normals_path = image_group.store_image(normals)

    # -------- LEFT --------
    @property
    def left_pixels(self):
        return self.pixels

    @property
    def left_metadata(self) -> imeta.ImageMetadata:
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
    def left_true_depth(self):
        """
        The left ground-truth depth image.
        This is the same as ground_truth_depth_data.
        :return: A numpy array, or None if no depth data is available.
        """
        return self.true_depth

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
    def left_camera_pose(self) -> Transform:
        """
        Get the underlying Transform object representing the pose of the left camera.
        This is useful to do things like convert points or poses to camera-relative.
        This is the same as 'camera_pose'
        :return: A Transform object
        """
        return self.camera_pose

    # -------- STEREO --------
    @property
    def stereo_offset(self) -> Transform:
        """
        The pose of the right camera relative to the left camera.
        Used for setting the stereo offset of systems
        :return:
        """
        if self.left_camera_pose is None:
            return None
        return self.left_camera_pose.find_relative(self.right_camera_pose)

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
                   image_group=left_image.image_group,
                   metadata=left_image.metadata,
                   right_metadata=right_image.metadata,
                   additional_metadata=additional_metadata,
                   depth=left_image.depth,
                   right_depth=right_image.depth,
                   normals=left_image.normals,
                   right_normals=right_image.normals)
