# Copyright (c) 2017, John Skinner
import typing
import enum
import bson
import numpy as np
import argus.util.transform as tf
import argus.util.dict_utils as du
import argus.util.database_helpers as dh
import argus.metadata.camera_intrinsics as cam_intr


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


class LabelledObject:
    """
    Metadata for a labelled object in an image.
    """
    def __init__(self, class_names: typing.Iterable[str], bounding_box: typing.Tuple[int, int, int, int],
                 label_color: typing.Tuple[int, int, int] = None, relative_pose: tf.Transform = None,
                 object_id: str = None):
        self._class_names = tuple(class_names)
        # Order is (upper left x, upper left y, width, height)
        self._bounding_box = (int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
        self._label_color = ((int(label_color[0]), int(label_color[1]), int(label_color[2]))
                             if label_color is not None else None)
        self._relative_pose = relative_pose
        self._object_id = object_id

    @property
    def class_names(self) -> typing.Tuple[str, ...]:
        return self._class_names

    @property
    def bounding_box(self) -> typing.Tuple[int, int, int, int]:
        """
        The bounding box, as (x, y, width, height).
        Location is measured from the top left of the image
        :return:
        """
        return self._bounding_box

    @property
    def label_color(self) -> typing.Tuple[int, int, int]:
        return self._label_color

    @property
    def relative_pose(self) -> tf.Transform:
        return self._relative_pose

    @property
    def object_id(self) -> str:
        return self._object_id

    def __eq__(self, other: typing.Any) -> bool:
        """
        Override equals. Labelled objects
        :param other:
        :return:
        """
        return (hasattr(other, 'class_names') and
                hasattr(other, 'bounding_box') and
                hasattr(other, 'label_color') and
                hasattr(other, 'relative_pose') and
                hasattr(other, 'object_id') and
                self.class_names == other.class_names and
                self.bounding_box == other.bounding_box and
                self.label_color == other.label_color and
                self.relative_pose == other.relative_pose and
                self.object_id == other.object_id)

    def __hash__(self) -> int:
        """
        Hash this object, so it can be in sets
        :return:
        """
        return hash((self.class_names, self.bounding_box, self.label_color, self.relative_pose, self.object_id))

    def serialize(self) -> dict:
        return {
            'class_names': list(self.class_names),
            'bounding_box': self.bounding_box,
            'label_color': self.label_color,
            'relative_pose': self.relative_pose.serialize() if self.relative_pose is not None else None,
            'object_id': self.object_id
        }

    @classmethod
    def deserialize(cls, serialized: dict) -> 'LabelledObject':
        kwargs = {}
        if 'class_names' in serialized:
            kwargs['class_names'] = tuple(serialized['class_names'])
        if 'bounding_box' in serialized:
            kwargs['bounding_box'] = (tuple(serialized['bounding_box'])
                                      if serialized['bounding_box'] is not None else None)
        if 'label_color' in serialized:
            kwargs['label_color'] = tuple(serialized['label_color']) if serialized['label_color'] is not None else None
        if 'relative_pose' in serialized:
            kwargs['relative_pose'] = (tf.Transform.deserialize(serialized['relative_pose'])
                                       if serialized['relative_pose'] is not None else None)
        if 'object_id' in serialized:
            kwargs['object_id'] = serialized['object_id']
        return cls(**kwargs)


class ImageMetadata:
    """
    A collection of metadata properties for images.
    There's a lot of properties here, not all of which are always available.

    Instances of this class are associated with Image objects
    """

    def __init__(self, source_type, hash_, camera_pose=None, right_camera_pose=None, intrinsics=None,
                 right_intrinsics=None, lens_focal_distance=None, aperture=None,
                 environment_type=None, light_level=None, time_of_day=None,
                 simulator=None, simulation_world=None, lighting_model=None, texture_mipmap_bias=None,
                 normal_maps_enabled=True, roughness_enabled=None, geometry_decimation=None,
                 procedural_generation_seed=None,
                 labelled_objects=None, average_scene_depth=None, base_image=None, transformation_matrix=None):
        self._hash = hash_

        self._source_type = ImageSourceType(source_type)
        self._environment_type = EnvironmentType(environment_type) if environment_type is not None else None
        self._light_level = LightingLevel(light_level) if light_level is not None else None
        self._time_of_day = TimeOfDay(time_of_day) if time_of_day is not None else None

        self._camera_pose = camera_pose
        self._right_camera_pose = right_camera_pose
        self._camera_intrinsics = intrinsics
        self._right_camera_intrinsics = right_intrinsics
        self._lens_focal_distance = float(lens_focal_distance) if lens_focal_distance is not None else None
        self._aperture = float(aperture) if aperture is not None else None

        # Computer graphics settings, for measuring image quality
        self._simulator = simulator
        self._simulation_world = str(simulation_world) if simulation_world is not None else None
        self._lighting_model = LightingModel(lighting_model) if lighting_model is not None else None
        self._texture_mipmap_bias = int(texture_mipmap_bias) if texture_mipmap_bias is not None else None
        self._normal_maps_enabled = bool(normal_maps_enabled) if normal_maps_enabled is not None else None
        self._roughness_enabled = bool(roughness_enabled) if roughness_enabled is not None else None
        self._geometry_decimation = float(geometry_decimation) if geometry_decimation is not None else None

        # Procedural Generation settings
        self._procedural_generation_seed = (int(procedural_generation_seed)
                                            if procedural_generation_seed is not None else None)

        # Labelling information
        self._labelled_objects = tuple(labelled_objects) if labelled_objects is not None else ()

        # Depth information
        self._average_scene_depth = float(average_scene_depth) if average_scene_depth is not None else None

        # Metadata from data augmentation and warping
        self._base_image = base_image
        self._affine_transformation_matrix = transformation_matrix

    def __eq__(self, other: typing.Any) -> bool:
        return (isinstance(other, ImageMetadata) and
                self.hash == other.hash and
                self.source_type == other.source_type and
                self.environment_type == other.environment_type and
                self.light_level == other.light_level and
                self.time_of_day == other.time_of_day and
                self.camera_pose == other.camera_pose and
                self.right_camera_pose == other.right_camera_pose and
                self.camera_intrinsics == other.camera_intrinsics and
                self.right_camera_intrinsics == other.right_camera_intrinsics and
                self.lens_focal_distance == other.lens_focal_distance and
                self.aperture == other.aperture and
                self.simulator == other.simulator and
                self.simulation_world == other.simulation_world and
                self.lighting_model == other.lighting_model and
                self.texture_mipmap_bias == other.texture_mipmap_bias and
                self.normal_maps_enabled == other.normal_maps_enabled and
                self.roughness_enabled == other.roughness_enabled and
                self.geometry_decimation == other.geometry_decimation and
                self.procedural_generation_seed == other.procedural_generation_seed and
                self.average_scene_depth == other.average_scene_depth and
                self.base_image == other.base_image and
                np.array_equal(self.affine_transformation_matrix, other.affine_transformation_matrix) and
                set(self.labelled_objects) == set(other.labelled_objects))

    def __hash__(self) -> int:
        return hash((self.hash, self.source_type, self.environment_type, self.light_level, self.time_of_day,
                     self.camera_pose, self.right_camera_pose, self.camera_intrinsics,
                     self.right_camera_intrinsics, self.lens_focal_distance, self.aperture,
                     self.simulator, self.simulation_world, self.lighting_model,
                     self.texture_mipmap_bias, self.normal_maps_enabled, self.roughness_enabled,
                     self.geometry_decimation, self.procedural_generation_seed, self.average_scene_depth,
                     hash(self.base_image), tuple(tuple(row) for row in self.affine_transformation_matrix)) +
                    tuple(hash(obj) for obj in self.labelled_objects))

    @property
    def hash(self) -> bytes:
        """
        The 64-bit xxhash of the image data.
        this is useful for quick comparisons of images,
        particularly within the database where we don't have the image data available.
        :return:
        """
        return self._hash

    @property
    def source_type(self) -> ImageSourceType:
        return self._source_type

    @property
    def environment_type(self) -> typing.Union[EnvironmentType, None]:
        return self._environment_type

    @property
    def light_level(self) -> typing.Union[LightingLevel, None]:
        return self._light_level

    @property
    def time_of_day(self) -> typing.Union[TimeOfDay, None]:
        return self._time_of_day

    @property
    def height(self) -> int:
        return self.camera_intrinsics.height

    @property
    def width(self) -> int:
        return self.camera_intrinsics.width

    @property
    def camera_pose(self) -> typing.Union[tf.Transform, None]:
        return self._camera_pose

    @property
    def right_camera_pose(self) -> typing.Union[tf.Transform, None]:
        return self._right_camera_pose

    @property
    def camera_intrinsics(self) -> cam_intr.CameraIntrinsics:
        """
        The camera intrinsics object, containing the image resolution, focal distances, and lens skew
        If you want them in matrix form, get the intrinsic_matrix property of this.
        :return:
        """
        return self._camera_intrinsics

    @property
    def right_camera_intrinsics(self) -> typing.Union[cam_intr.CameraIntrinsics, None]:
        """
        If this is a stereo image, a object containing the right camera intrinsics, fx, fy, cx, cy.
        If you want them in matrix form, use right_intrinsic_matrix.
        You can also use this to get the dimensions of the right image.
        :return:
        """
        return self._right_camera_intrinsics

    @property
    def lens_focal_distance(self) -> typing.Union[float, None]:
        """
        The distance to the in-focus field of the camera lens.
        The distance from the camera at which an object is in focus.
        This is distinct from the length of the pinhole camera, although the values are often the same.
        :return:
        """
        return self._lens_focal_distance

    @property
    def aperture(self) -> float:
        return self._aperture

    @property
    def simulator(self) -> typing.Union[None, bson.ObjectId]:
        return self._simulator

    @property
    def simulation_world(self) -> typing.Union[str, None]:
        return self._simulation_world

    @property
    def lighting_model(self) -> LightingModel:
        return self._lighting_model

    @property
    def texture_mipmap_bias(self) -> int:
        return self._texture_mipmap_bias

    @property
    def normal_maps_enabled(self) -> bool:
        return self._normal_maps_enabled

    @property
    def roughness_enabled(self) -> typing.Union[bool, None]:
        return self._roughness_enabled

    @property
    def geometry_decimation(self) -> typing.Union[float, None]:
        return self._geometry_decimation

    @property
    def procedural_generation_seed(self):
        return self._procedural_generation_seed

    @property
    def labelled_objects(self):
        return self._labelled_objects

    @property
    def average_scene_depth(self):
        """
        The approximate average depth o the scene.
        This gives a sense of whether the image is a wide open landscape,
        or a close-in shot
        :return: A float for the average scene depth, or for the average of many images,
        """
        return self._average_scene_depth

    @property
    def base_image(self):
        """
        If this image is warped from another image, this is the source image.
        Otherwise, None
        :return: The base image this warped image was produced from, or None.
        """
        return self._base_image

    @property
    def affine_transformation_matrix(self):
        """
        If this image is a warped image, this is the affine transformation matrix used to
        warp the original image to produce this one.
        :return:
        """
        return self._affine_transformation_matrix

    def clone(self, **kwargs):
        """
        Clone the metadata, optionally changing some of the values.
        This is the best way to create a new metadata based on an existing one
        :param kwargs: Overridden arguments for differences to the cloned metadata, same as the constructor arguments
        :return: a new image metadata object
        """
        du.defaults(kwargs, {
            'hash_': self.hash,
            'source_type': self.source_type,
            'camera_pose': self.camera_pose,
            'right_camera_pose': self.right_camera_pose,
            'intrinsics': self.camera_intrinsics,
            'right_intrinsics': self.right_camera_intrinsics,
            'environment_type': self.environment_type,
            'light_level': self.light_level,
            'time_of_day': self.time_of_day,
            'lens_focal_distance': self.lens_focal_distance,
            'aperture': self.aperture,
            'simulator': self.simulator,
            'simulation_world': self.simulation_world,
            'lighting_model': self.lighting_model,
            'texture_mipmap_bias': self.texture_mipmap_bias,
            'normal_maps_enabled': self.normal_maps_enabled,
            'roughness_enabled': self.roughness_enabled,
            'geometry_decimation': self.geometry_decimation,
            'procedural_generation_seed': self.procedural_generation_seed,
            'labelled_objects': tuple((LabelledObject(
                class_names=obj.class_names,
                bounding_box=obj.bounding_box,
                label_color=obj.label_color,
                relative_pose=obj.relative_pose,
                object_id=obj.object_id)
                for obj in self.labelled_objects)),
            'average_scene_depth': self.average_scene_depth,
            'base_image': self.base_image,
            'transformation_matrix': self.affine_transformation_matrix
        })
        return ImageMetadata(**kwargs)

    def serialize(self):
        serialized = {
            'hash': self.hash,
            'source_type': self.source_type.value,
            'environment_type': self.environment_type.value if self.environment_type is not None else None,
            'light_level': self.light_level.value if self.light_level is not None else None,
            'time_of_day': self.time_of_day.value if self.time_of_day is not None else None,

            'camera_pose': self.camera_pose.serialize() if self.camera_pose is not None else None,
            'right_camera_pose': self.right_camera_pose.serialize() if self.right_camera_pose is not None else None,
            'intrinsics': self.camera_intrinsics.serialize() if self.camera_intrinsics is not None else None,
            'right_intrinsics': (self.right_camera_intrinsics.serialize()
                                 if self.right_camera_intrinsics is not None else None),
            'lens_focal_distance': self.lens_focal_distance,
            'aperture': self.aperture,

            'simulator': self.simulator,
            'simulation_world': self.simulation_world,
            'lighting_model': self.lighting_model.value if self.lighting_model is not None else None,
            'texture_mipmap_bias': self.texture_mipmap_bias,
            'normal_maps_enabled': self.normal_maps_enabled,
            'roughness_enabled': self.roughness_enabled,
            'geometry_decimation': self._geometry_decimation,

            'procedural_generation_seed': self.procedural_generation_seed,

            'labelled_objects': [obj.serialize() for obj in self.labelled_objects],

            'average_scene_depth': self.average_scene_depth,
            'base_image': self.base_image,
            'transformation_matrix': self.affine_transformation_matrix
        }
        dh.add_schema_version(serialized, 'metadata:image_metadata:ImageMetadata', 1)
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        kwargs = {}
        update_schema(serialized)
        direct_copy_keys = ['source_type', 'environment_type', 'light_level', 'time_of_day',
                            'lens_focal_distance', 'aperture', 'simulator', 'simulation_world', 'lighting_model',
                            'texture_mipmap_bias', 'normal_maps_enabled', 'roughness_enabled', 'geometry_decimation',
                            'procedural_generation_seed', 'average_scene_depth', 'base_image', 'transformation_matrix']
        for key in direct_copy_keys:
            if key in serialized:
                kwargs[key] = serialized[key]
        if 'hash' in serialized:
            kwargs['hash_'] = serialized['hash']
        if 'camera_pose' in serialized and serialized['camera_pose'] is not None:
            kwargs['camera_pose'] = tf.Transform.deserialize(serialized['camera_pose'])
        if 'right_camera_pose' in serialized and serialized['right_camera_pose'] is not None:
            kwargs['right_camera_pose'] = tf.Transform.deserialize(serialized['right_camera_pose'])
        if 'intrinsics' in serialized and serialized['intrinsics'] is not None:
            kwargs['intrinsics'] = cam_intr.CameraIntrinsics.deserialize(serialized['intrinsics'])
        if 'right_intrinsics' in serialized and serialized['right_intrinsics'] is not None:
            kwargs['right_intrinsics'] = cam_intr.CameraIntrinsics.deserialize(serialized['right_intrinsics'])
        if 'labelled_objects' in serialized:
            kwargs['labelled_objects'] = tuple(LabelledObject.deserialize(s_obj)
                                               for s_obj in serialized['labelled_objects'])
        return cls(**kwargs)


def update_schema(serialized: dict):
    """
    Update the serialized image metadata to the latest version.
    :param serialized:
    :return:
    """
    version = dh.get_schema_version(serialized, 'metadata:image_metadata:ImageMetadata')
    if version < 1:
        # unversioned -> version 1
        if 'width' in serialized:
            if 'intrinsics' in serialized and isinstance(serialized['intrinsics'], dict):
                serialized['intrinsics']['width'] = serialized['width']
            if 'right_intrinsics' in serialized and isinstance(serialized['right_intrinsics'], dict):
                serialized['right_intrinsics']['width'] = serialized['width']
        if 'height' in serialized:
            if 'intrinsics' in serialized and isinstance(serialized['intrinsics'], dict):
                serialized['intrinsics']['height'] = serialized['height']
            if 'right_intrinsics' in serialized and isinstance(serialized['right_intrinsics'], dict):
                serialized['right_intrinsics']['height'] = serialized['height']
        if 'focal_distance' in serialized:
            serialized['lens_focal_distance'] = serialized['focal_distance']


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
        hash_=left_image_metadata.hash,
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
