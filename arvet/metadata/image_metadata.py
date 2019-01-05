# Copyright (c) 2017, John Skinner
import typing
import enum
import numpy as np
import pymodm
from arvet.database.image_field import ImageField
from arvet.database.transform_field import TransformField
from arvet.database.enum_field import EnumField
import arvet.metadata.camera_intrinsics as cam_intr


class ImageSourceType(enum.Enum):
    REAL_WORLD = 0
    SYNTHETIC = 1


class EnvironmentType(enum.Enum):
    INDOOR_CLOSE = 0
    INDOOR = 1
    OUTDOOR_URBAN = 2
    OUTDOOR_LANDSCAPE = 3


class LightingLevel(enum.Enum):
    PITCH_BLACK = 0
    POOR = 1
    DIM = 2
    EVENLY_LIT = 3
    WELL_LIT = 4
    BRIGHT = 5


class LightingModel(enum.Enum):
    UNLIT = 0
    LIT = 1


class TimeOfDay(enum.Enum):
    NIGHT = 0
    DAWN = 1
    MORNING = 2
    DAY = 3
    AFTERNOON = 4
    TWILIGHT = 5


class LabelledObject(pymodm.EmbeddedMongoModel):
    """
    Metadata for a labelled object in an image.
    TODO: More properties, such as: sobel strength
    """

    class_names = pymodm.fields.ListField(pymodm.fields.CharField(), required=True)
    x = pymodm.fields.IntegerField(required=True)
    y = pymodm.fields.IntegerField(required=True)
    width = pymodm.fields.IntegerField(required=True)
    height = pymodm.fields.IntegerField(required=True)
    relative_pose = TransformField()
    instance_name = pymodm.fields.CharField()

    @property
    def bounding_box(self) -> typing.Tuple[int, int, int, int]:
        """
        The bounding box, as (x, y, width, height).
        Location is measured from the top left of the image
        :return:
        """
        return self.x, self.y, self.width, self.height

    @property
    def centroid(self) -> typing.Tuple[float, float]:
        """
        The centroid of the object in the image. May be a fractional pixel like 13.5
        :return:
        """
        return self.x + self.width / 2, self.y + self.height / 2

    def __eq__(self, other: typing.Any) -> bool:
        """
        Override equals. Labelled objects
        :param other:
        :return:
        """
        return (hasattr(other, 'class_names') and
                hasattr(other, 'bounding_box') and
                hasattr(other, 'relative_pose') and
                hasattr(other, 'instance_name') and
                self.class_names == other.class_names and
                self.bounding_box == other.bounding_box and
                self.relative_pose == other.relative_pose and
                self.instance_name == other.instance_name)

    def __hash__(self):
        return hash(self._get_hash_tuple())

    def _get_hash_tuple(self):
        return (
            self.x,
            self.y,
            self.width,
            self.height,
            self.relative_pose,
            self.instance_name
        ) + tuple(self.class_names)


class MaskedObject(LabelledObject):
    """
    More specific metadata for when we know the pixel segmentation for an object
    """
    mask = ImageField(required=True)

    def __init__(self, *args, **kwargs):
        # Validate and ensure that the mask image match the width and height attribute
        mask = args[8] if len(args) >= 9 else kwargs.get('mask', None)
        if mask is not None:
            width = args[3] if len(args) >= 4 else kwargs.get('width', None)
            height = args[4] if len(args) >= 5 else kwargs.get('height', None)
            if width is not None and width != mask.shape[1]:
                raise ValueError("Mask width does not match width argument")
            elif width is None:
                kwargs['width'] = mask.shape[1]
            if height is not None and height != mask.shape[0]:
                raise ValueError("Mask height does not match height argument")
            elif height is None:
                kwargs['height'] = mask.shape[0]
        super().__init__(*args, **kwargs)

    def __eq__(self, other: typing.Any) -> bool:
        """
        Override equals. Labelled objects
        :param other:
        :return:
        """
        return (super().__eq__(other) and
                hasattr(other, 'mask') and
                np.array_equal(self.mask, other.mask))

    def _get_hash_tuple(self):
        return super()._get_hash_tuple() + (self.mask.data.tobytes(),)


def calculate_edge_strength(object_image, mask):
    # TODO: Use sobel edge detectors for each channel, then build the matrix:
    # s = [[Rx^2 + Gx^2 + Bx^2, RxRy + GxGy +BxBy], [RxRy + GxGy +BxBy, Ry^2 + Gy^2 + By^2]]
    # The edge strength is the largest eigenvector,
    # if T is a + d, and D is ad - bc
    # L = T / 2 + sqrt(T^2/4 - D)
    # There's probably an analytical solution, but it's nasty
    # T = Rx^2 + Gx^2 + Bx^2 + Ry^2 + Gy^2 + By^2
    # D = Rx^2 Gy^2 + Rx^2 By^2 + Gx^2 Ry^2 + Gx^2 By^2 + Bx^2 Ry^2 + Bx^2 Gy^2 - 2 RxGxRyGy - 2 RxBxRyBy - 2 GxBxGyBy
    pass


class ImageMetadata(pymodm.EmbeddedMongoModel):
    """
    A collection of metadata properties for images.
    There's a lot of properties here, not all of which are always available.

    Instances of this class are associated with Image objects
    """
    img_hash = pymodm.BinaryField(required=True)
    source_type = EnumField(ImageSourceType, required=True)

    camera_pose = TransformField()
    right_camera_pose = TransformField()
    intrinsics = pymodm.fields.EmbeddedDocumentField(cam_intr.CameraIntrinsics)
    right_intrinsics = pymodm.fields.EmbeddedDocumentField(cam_intr.CameraIntrinsics)
    lens_focal_distance = pymodm.fields.FloatField()
    aperture = pymodm.fields.FloatField()

    red_mean = pymodm.fields.FloatField()
    red_std = pymodm.fields.FloatField()
    green_mean = pymodm.fields.FloatField()
    green_std = pymodm.fields.FloatField()
    blue_mean = pymodm.fields.FloatField()
    blue_std = pymodm.fields.FloatField()
    depth_mean = pymodm.fields.FloatField()
    depth_std = pymodm.fields.FloatField()

    environment_type = EnumField(EnvironmentType)
    light_level = EnumField(LightingLevel)
    time_of_day = EnumField(TimeOfDay)

    # TODO: Reference a simulator?
    # _simulator = pymodm.fields.ReferenceField(Simulator, on_delete=pymodm.fields.ReferenceField.NULLIFY)
    simulation_world = pymodm.fields.CharField()
    lighting_model = EnumField(LightingModel)
    texture_mipmap_bias = pymodm.fields.IntegerField()
    normal_maps_enabled = pymodm.fields.BooleanField()
    roughness_enabled = pymodm.fields.BooleanField()
    geometry_decimation = pymodm.fields.FloatField()

    procedural_generation_seed = pymodm.fields.IntegerField()
    labelled_objects = pymodm.fields.ListField(pymodm.EmbeddedDocumentField(LabelledObject))

    def __eq__(self, other: typing.Any) -> bool:
        return (isinstance(other, ImageMetadata) and
                self.img_hash == other.img_hash and
                self.source_type == other.source_type and

                self.camera_pose == other.camera_pose and
                self.right_camera_pose == other.right_camera_pose and
                self.intrinsics == other.intrinsics and
                self.right_intrinsics == other.right_intrinsics and
                self.lens_focal_distance == other.lens_focal_distance and
                self.aperture == other.aperture and

                self.red_mean == other.red_mean and
                self.red_std == other.red_std and
                self.green_mean == other.green_mean and
                self.green_std == other.green_std and
                self.blue_mean == other.blue_mean and
                self.blue_std == other.blue_std and
                self.depth_mean == other.depth_mean and
                self.depth_std == other.depth_std and

                self.environment_type == other.environment_type and
                self.light_level == other.light_level and
                self.time_of_day == other.time_of_day and

                self.simulation_world == other.simulation_world and
                self.lighting_model == other.lighting_model and
                self.texture_mipmap_bias == other.texture_mipmap_bias and
                self.normal_maps_enabled == other.normal_maps_enabled and
                self.roughness_enabled == other.roughness_enabled and
                self.geometry_decimation == other.geometry_decimation and

                self.procedural_generation_seed == other.procedural_generation_seed and
                set(self.labelled_objects) == set(other.labelled_objects))

    def __hash__(self) -> int:
        return hash(
            (
                self.img_hash,
                self.source_type,

                self.camera_pose,
                self.right_camera_pose,
                self.intrinsics,
                self.right_intrinsics,
                self.lens_focal_distance,
                self.aperture,

                self.red_mean,
                self.red_std,
                self.green_mean,
                self.green_std,
                self.blue_mean,
                self.blue_std,
                self.depth_mean,
                self.depth_std,

                self.environment_type,
                self.light_level,
                self.time_of_day,

                self.simulation_world,
                self.lighting_model,
                self.texture_mipmap_bias,
                self.normal_maps_enabled,
                self.roughness_enabled,
                self.geometry_decimation,
                self.procedural_generation_seed
            ) + tuple(hash(obj) for obj in self.labelled_objects)
        )


T = typing.TypeVar


def merge(*args: T) -> T:
    """
    Merge a list of things, returning the first non-null value.
    Returns None if all the arguments are None.
    This is used to merge metadata values from multiple metadata objects, treating None as unspecified.
    :param args:
    :return: The first non-zero argument
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


def merge_stereo(left_image_metadata: ImageMetadata, right_image_metadata: ImageMetadata) -> ImageMetadata:
    """
    Merge two image metadata to produce the metadata for a stereo image.
    Mostly uses the left image values where available, and falls back to the right image where they are missing.
    :param left_image_metadata: The metadata for the left image
    :param right_image_metadata: The metadata for the right image
    :return:
    """
    return ImageMetadata(
        img_hash=left_image_metadata.img_hash,
        source_type=left_image_metadata.source_type,

        camera_pose=left_image_metadata.camera_pose,
        right_camera_pose=right_image_metadata.camera_pose,
        intrinsics=left_image_metadata.camera_intrinsics,
        right_intrinsics=right_image_metadata.camera_intrinsics,
        lens_focal_distance=merge(left_image_metadata.lens_focal_distance, right_image_metadata.lens_focal_distance),
        aperture=merge(left_image_metadata.aperture, right_image_metadata.aperture),

        environment_type=merge(left_image_metadata.environment_type, right_image_metadata.environment_type),
        light_level=merge(left_image_metadata.light_level, right_image_metadata.light_level),
        time_of_day=merge(left_image_metadata.time_of_day, right_image_metadata.time_of_day),
        simulator=merge(left_image_metadata.simulator, right_image_metadata.simulator),
        simulation_world=merge(left_image_metadata.simulation_world, right_image_metadata.simulation_world),
        lighting_model=merge(left_image_metadata.lighting_model, right_image_metadata.lighting_model),
        texture_mipmap_bias=merge(left_image_metadata.texture_mipmap_bias, right_image_metadata.texture_mipmap_bias),
        normal_maps_enabled=merge(left_image_metadata.normal_maps_enabled, right_image_metadata.normal_maps_enabled),
        roughness_enabled=merge(left_image_metadata.roughness_enabled, right_image_metadata.roughness_enabled),
        geometry_decimation=merge(left_image_metadata.geometry_decimation, right_image_metadata.geometry_decimation),
        procedural_generation_seed=merge(left_image_metadata.procedural_generation_seed,
                                         right_image_metadata.procedural_generation_seed),
        labelled_objects=merge(left_image_metadata.labelled_objects, right_image_metadata.labelled_objects),
        average_scene_depth=merge(left_image_metadata.average_scene_depth, right_image_metadata.average_scene_depth),
        base_image=merge(left_image_metadata.base_image, right_image_metadata.base_image),
        transformation_matrix=merge(left_image_metadata.affine_transformation_matrix,
                                    right_image_metadata.affine_transformation_matrix)
    )
