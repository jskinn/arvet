#Copyright (c) 2017, John Skinner
import cv2
import util.dict_utils as du
import systems.feature.feature_detector


class ORBDetector(systems.feature.feature_detector.FeatureDetector):
    """
    A FeatureDetector that uses ORB features.
    ORB settings are _really_ finicky, a lot of combinations will cause errors.
    I don't have a good handle on what combinations work at all
    """

    def __init__(self, config=None, id_=None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        if config is None:
            config = {}
        config = du.defaults({}, config, kwargs, {
            'num_features': 1000,
            'scale_factor': 1.2,
            'num_levels': 8,
            'edge_threshold': 31,
            'patch_size': 31,
            'fast_threshold': 20
        })
        self._num_features = int(config['num_features'])
        self._scale_factor = max(float(config['scale_factor']), 1.0)
        self._num_levels = int(config['num_levels'])
        self._edge_threshold = int(config['edge_threshold'])
        self._patch_size = int(config['patch_size'])
        self._fast_threshold = int(config['fast_threshold'])

        # variables for trial state
        self._orb = None
        self._key_points = None

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, number_of_features):
        if not self.is_trial_running():
            self._num_features = int(number_of_features)

    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, scale_factor):
        if not self.is_trial_running():
            self._scale_factor = float(scale_factor)

    @property
    def number_of_levels(self):
        return self._num_levels

    @number_of_levels.setter
    def number_of_levels(self, number_of_levels):
        if not self.is_trial_running():
            self._num_levels = int(number_of_levels)

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size):
        if not self.is_trial_running():
            self._patch_size = int(patch_size)

    @property
    def edge_threshold(self):
        return self._edge_threshold

    @edge_threshold.setter
    def edge_threshold(self, threshold):
        if not self.is_trial_running():
            self._edge_threshold = float(threshold)

    @property
    def fast_threshold(self):
        return self._fast_threshold

    @fast_threshold.setter
    def fast_threshold(self, fast_threshold):
        if not self.is_trial_running():
            self._fast_threshold = float(fast_threshold)

    def make_detector(self):
        """
        Make the SIFT detector for this system.
        :return:
        """
        return cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=self.scale_factor,
            nlevels=self.number_of_levels,
            edgeThreshold=self.edge_threshold,
            patchSize=self.patch_size,
            fastThreshold=self.fast_threshold
        )

    def get_system_settings(self):
        """
        Get the settings values used by the detector
        :return:
        """
        return {
            'num_features': self.num_features,
            'scale_factor': self.scale_factor,
            'num_levels': self.number_of_levels,
            'edge_threshold': self.edge_threshold,
            'patch_size': self.patch_size,
            'fast_threshold': self.fast_threshold
        }

    def validate(self):
        """
        Check that the entity is valid.
        This lets us check things like whether references files still exist.
        :return: True iff the entity is in a valid state, and can be used.
        """
        # I can't think of a way for this object to be invalid
        return True

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        serialized = super().serialize()
        serialized['config'] = self.get_system_settings()
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        """
        Deserialize an Entity as retrieved from the database
        Also accepts keyword arguments that are passed directly to the entity constructor,
        so that we can have required parameters to the initialization
        :param serialized_representation: dict
        :param db_client: DatabaseClient for deserializing child objects
        :return: An instance of the entity class, constructed from the serialized representation
        """
        if 'config' in serialized_representation:
            kwargs['config'] = serialized_representation['config']
        return super().deserialize(serialized_representation, db_client, **kwargs)
