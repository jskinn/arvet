#Copyright (c) 2017, John Skinner
import metadata.image_metadata as imeta


class ImageMetadataSummaryBuilder:
    # TODO: Create a builder class that amalgamates many image metadata into a summary.

    def __init__(self):
        pass

    def add(self, image_metadata):
        pass


class ImageMetadataSummary:
    """
    A summary class for many image metadata.
    Is a combination metadata, that counts how often different
    values for each property occur.

    This class is associated with results objects, summarizing the kinds of images
    the test was performed using.
    """

    def __init__(self):
        self._pose = []
        self._source_type = imeta.ImageSourceType.REAL_WORLD
        self._environment_type = imeta.EnvironmentType.INDOOR_CLOSE
        self._light_level = imeta.LightingLevel.EVENLY_LIT
        self._time_of_day = imeta.TimeOfDay.NOT_APPLICABLE

        self._height = 0
        self._width = 0
        self._fov = 90.0
        self._focal_length = 1000.0
        self._aperature = 22

        # Computer graphics settings, for measuring image quality
        self._simulation_world = None
        self._lighting_model = None
        self._texture_mipmap_bias = None
        self._normal_mipmap_bias = None
        self._roughness_enabled = None
        self._geometry_decimation = None
        self._depth_of_field = None

        # Labelling information
        self._label_classes = []
        self._label_bounding_boxes = None
        self._distance_to_labelled_objects = None

        # Depth information
        self._average_scene_depth = None
