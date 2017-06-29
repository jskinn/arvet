import enum
import util.dict_utils as du
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


class BoundingBox:
    """
    A bounding box.
    x and y are column-row of the top left corner, height and width extending down and right from there.
    Origin is top left corner of the image.
    Bounding boxes should be mapped to a particular image,
    and image coordinates are relative to the base resolution of that image.
    DEPRECATED
    """
    def __init__(self, class_name, confidence, x, y, height, width):
        self.class_name = class_name
        self.confidence = float(confidence)
        self.x = int(x)
        self.y = int(y)
        self.height = int(height)
        self.width = int(width)

    def __eq__(self, other):
        """
        Override equals. Bounding boxes are equal if the have the same class, confidence, and shape
        :param other:
        :return:
        """
        return (hasattr(other, 'class_name') and
                hasattr(other, 'confidence') and
                hasattr(other, 'x') and
                hasattr(other, 'y') and
                hasattr(other, 'height') and
                hasattr(other, 'width') and
                self.class_name == other.class_name and
                self.confidence == other.confidence and
                self.x == other.x and
                self.y == other.y and
                self.height == other.height and
                self.width == other.width)

    def __hash__(self):
        """
        Hash this object, so it can be in sets
        :return:
        """
        return hash((self.class_name, self.confidence, self.x, self.y, self.height, self.width))

    def serialize(self):
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'x': self.x,
            'y': self.y,
            'height': self.height,
            'width': self.width
        }

    @classmethod
    def deserialize(cls, serialized):
        du.defaults(serialized, {
            'class_name': 'bg',
            'confidence': 0,
            'x': 0,
            'y': 0,
            'height': 0,
            'width': 0
        }, modify_base=True)
        return cls(**serialized)


class LabelledObject:
    """
    Metadata for a labelled object in an image.
    """
    def __init__(self, class_names, bounding_box, label_color=None, relative_pose=None, object_id=None):
        self._class_names = tuple(class_names)
        self._bounding_box = (int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
        self._label_color = label_color
        self._relative_pose = relative_pose
        self._object_id = object_id

    @property
    def class_names(self):
        return self._class_names

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def label_color(self):
        return self._label_color

    @property
    def relative_pose(self):
        return self._relative_pose

    @property
    def object_id(self):
        return self._object_id

    def __eq__(self, other):
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

    def __hash__(self):
        """
        Hash this object, so it can be in sets
        :return:
        """
        return hash((self.class_names, self.bounding_box, self.label_color, self.relative_pose, self.object_id))

    def serialize(self):
        return {
            'class_names': list(self.class_names),
            'bounding_box': self.bounding_box,
            'label_color': self.label_color,
            'relative_pose': self.relative_pose.serialize() if self.relative_pose is not None else None,
            'object_id': self.object_id
        }

    @classmethod
    def deserialize(cls, serialized):
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

    def __init__(self, source_type, height, width, environment_type=None, light_level=None, time_of_day=None, fov=None,
                 focal_length=None, aperture=None, simulation_world=None, lighting_model=None, texture_mipmap_bias=None,
                 normal_mipmap_bias=None, roughness_enabled=None, geometry_decimation=None,
                 procedural_generation_seed=None, labelled_objects=None, average_scene_depth=None):
        self._source_type = ImageSourceType(source_type)
        self._environment_type = EnvironmentType(environment_type) if environment_type is not None else None
        self._light_level = LightingLevel(light_level) if light_level is not None else None
        self._time_of_day = TimeOfDay(time_of_day) if time_of_day is not None else None

        self._height = int(height)
        self._width = int(width)
        self._fov = float(fov) if fov is not None else None
        self._focal_length = float(focal_length) if focal_length is not None else None
        self._aperture = float(aperture) if aperture is not None else None

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
        self._labelled_objects = tuple(labelled_objects) if labelled_objects is not None else ()

        # Depth information
        self._average_scene_depth = float(average_scene_depth) if average_scene_depth is not None else None

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

    def serialize(self):
        return {
            'source_type': self.source_type.value,
            'environment_type': self.environment_type.value if self.environment_type is not None else None,
            'light_level': self.light_level.value if self.light_level is not None else None,
            'time_of_day': self.time_of_day.value if self.time_of_day is not None else None,

            'height': self.height,
            'width': self.width,
            'fov': self.fov,
            'focal_length': self.focal_length,
            'aperture': self.aperture,

            'simulation_world': self.simulation_world,
            'lighting_model': self.lighting_model.value if self.lighting_model is not None else None,
            'texture_mipmap_bias': self.texture_mipmap_bias,
            'normal_mipmap_bias': self.normal_mipmap_bias,
            'roughness_enabled': self.roughness_enabled,
            'geometry_decimation': self._geometry_decimation,

            'procedural_generation_seed': self.procedural_generation_seed,

            'labelled_objects': [obj.serialize() for obj in self.labelled_objects],

            'average_scene_depth': self.average_scene_depth
        }

    @classmethod
    def deserialize(cls, serialized):
        kwargs = {}
        direct_copy_keys = ['source_type', 'environment_type', 'light_level', 'time_of_day', 'height', 'width', 'fov',
                            'focal_length', 'aperture', 'simulation_world', 'lighting_model', 'texture_mipmap_bias',
                            'normal_mipmap_bias', 'roughness_enabled', 'geometry_decimation',
                            'procedural_generation_seed', 'average_scene_depth']
        for key in direct_copy_keys:
            if key in serialized:
                kwargs[key] = serialized[key]
        if 'labelled_objects' in serialized:
            kwargs['labelled_objects'] = (LabelledObject.deserialize(s_obj)
                                          for s_obj in serialized['labelled_objects'])
        return cls(**kwargs)
