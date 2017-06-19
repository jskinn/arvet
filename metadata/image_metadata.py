import enum
import util.transform as tf


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


class ImageMetadata:
    """
    A collection of metadata properties for images.
    There's a lot of properties here, not all of which are always available.

    Instances of this class are associated with Image objects
    """

    def __init__(self, pose, source_type, environment_type, light_level, time_of_day, height, width, fov, focal_length,
                 aperture, simulation_world=None, lighting_model=None, texture_mipmap_bias=None,
                 normal_mipmap_bias=None, roughness_enabled=None, geometry_decimation=None,
                 procedural_generation_seed=None, label_classes=None, label_bounding_boxes=None,
                 distances_to_labelled_objects=None, average_scene_depth=None):
        if isinstance(pose, tf.Transform):
            self._pose = pose
        else:
            self._pose = tf.Transform(pose)
        self._source_type = ImageSourceType(source_type)
        self._environment_type = EnvironmentType(environment_type)
        self._light_level = LightingLevel(light_level)
        self._time_of_day = TimeOfDay(time_of_day)

        self._height = int(height)
        self._width = int(width)
        self._fov = float(fov)
        self._focal_length = float(focal_length)
        self._aperture = float(aperture)

        # Computer graphics settings, for measuring image quality
        self._simulation_world = str(simulation_world) if simulation_world is not None else None
        self._lighting_model = LightingModel(lighting_model) if lighting_model is not None else None
        self._texture_mipmap_bias = int(texture_mipmap_bias) if texture_mipmap_bias is not None else None
        self._normal_mipmap_bias = int(normal_mipmap_bias) if normal_mipmap_bias is not None else None
        self._roughness_enabled = bool(roughness_enabled) if roughness_enabled is not None else None
        self._geometry_decimation = int(geometry_decimation) if geometry_decimation is not None else None

        # Procedural Generation settings
        self._procedural_generation_seed = (int(procedural_generation_seed)
                                            if procedural_generation_seed is not None else None)

        # Labelling information
        self._label_classes = list(label_classes)
        self._label_bounding_boxes = list(label_bounding_boxes)
        self._distances_to_labelled_objects = dict(distances_to_labelled_objects)

        # Depth information
        self._average_scene_depth = float(average_scene_depth)

    @property
    def pose(self):
        return self._pose

    @property
    def source_type(self):
        return self._source_type

    @property
    def environment_type(self):
        return self._environment_type

    @property
    def light_level(self):
        return self._light_level

    @property
    def time_of_day(self):
        return self._time_of_day

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def fov(self):
        return self._fov

    @property
    def focal_length(self):
        return self._focal_length

    @property
    def aperture(self):
        return self._aperture

    @property
    def simulation_world(self):
        return self._simulation_world

    @property
    def lighting_model(self):
        return self._lighting_model

    @property
    def texture_mipmap_bias(self):
        return self._texture_mipmap_bias

    @property
    def normal_mipmap_bias(self):
        return self._normal_mipmap_bias

    @property
    def roughness_enabled(self):
        return self._roughness_enabled

    @property
    def geometry_decimation(self):
        return self._geometry_decimation

    @property
    def procedural_generation_seed(self):
        return self._procedural_generation_seed

    @property
    def label_classes(self):
        return self._label_classes

    @property
    def label_bounding_boxes(self):
        return self._label_bounding_boxes

    @property
    def distances_to_labelled_objects(self):
        return self._distances_to_labelled_objects

    @property
    def average_scene_depth(self):
        """
        The approximate average depth o the scene.
        This gives a sense of whether the image is a wide open landscape,
        or a close-in shot
        :return: A float for the average scene depth, or for the average of many images,
        """
        return self._average_scene_depth

    def serialize(self):
        return {
            'pose': self.pose.serialize(),
            'source_type': self.source_type.value,
            'environment_type': self.environment_type.value,
            'light_level': self.light_level.value,
            'time_of_day': self.time_of_day.value,

            'height': self.height,
            'width': self.width,
            'fov': self.fov,
            'focal_length': self.focal_length,
            'aperture': self.aperture,

            'simulation_world': self.simulation_world,
            'lighting_model': self.lighting_model.value,
            'texture_mipmap_bias': self.texture_mipmap_bias,
            'normal_mipmap_bias': self.normal_mipmap_bias,
            'roughness_enabled': self.roughness_enabled,
            'geometry_decimation': self._geometry_decimation,

            'procedural_generation_seed': self.procedural_generation_seed,

            'label_classes': self.label_classes,
            'label_bounding_boxes': self.label_bounding_boxes,
            'distances_to_labelled_objects': self._distances_to_labelled_objects,

            'average_scene_depth': self.average_scene_depth
        }

    @classmethod
    def deserialize(cls, serialized):
        kwargs = {}
        if 'pose' in serialized:
            kwargs['pose'] = tf.Transform.deserialize(serialized['pose'])
        direct_copy_keys = ['source_type', 'environment_type', 'light_level', 'time_of_day', 'height', 'width', 'fov',
                            'focal_length', 'aperture', 'simulation_world', 'lighting_model', 'texture_mipmap_bias',
                            'normal_mipmap_bias', 'roughness_enabled', 'geometry_decimation',
                            'procedural_generation_seed', 'label_classes', 'label_bounding_boxes',
                            'distances_to_labelled_objects', 'average_scene_depth']
        for key in direct_copy_keys:
            if key in serialized:
                kwargs[key] = serialized[key]

        return cls(**kwargs)
