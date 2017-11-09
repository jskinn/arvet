# Copyright (c) 2017, John Skinner
import cv2
import cv2.xfeatures2d
import util.dict_utils as du
import systems.feature.feature_detector


class SiftDetector(systems.feature.feature_detector.FeatureDetector):
    """
    A FeatureDetector that uses SIFT features
    """

    def __init__(self, config=None, id_=None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        if config is None:
            config = {}
        config = du.defaults(config, kwargs, {
            'num_features': 0,
            'num_octave_layers': 4,
            'contrast_threshold': 0.04,
            'edge_threshold': 10,
            'sigma': 1.6
        }, modify_base=False)

        self._num_features = int(config['num_features'])
        self._num_octave_layers = int(config['num_octave_layers'])
        self._contrast_threshold = float(config['contrast_threshold'])
        self._edge_threshold = float(config['edge_threshold'])
        self._sigma = float(config['sigma'])

        # variables for trial state
        self._sift = None
        self._key_points = None

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, number_of_features):
        if not self.is_trial_running():
            self._num_features = int(number_of_features)

    @property
    def num_octave_layers(self):
        return self._num_octave_layers

    @num_octave_layers.setter
    def num_octave_layers(self, number_of_octave_layers):
        if not self.is_trial_running():
            self._num_octave_layers = int(number_of_octave_layers)

    @property
    def contrast_threshold(self):
        return self._contrast_threshold

    @contrast_threshold.setter
    def contrast_threshold(self, threshold):
        if not self.is_trial_running():
            self._contrast_threshold = float(threshold)

    @property
    def edge_threshold(self):
        return self._edge_threshold

    @edge_threshold.setter
    def edge_threshold(self, threshold):
        if not self.is_trial_running():
            self._edge_threshold = float(threshold)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if not self.is_trial_running():
            self._sigma = float(sigma)

    def make_detector(self):
        """
        Make the SIFT detector for this system.
        :return:
        """
        return cv2.xfeatures2d.SIFT_create(nfeatures=self.num_features,
                                           nOctaveLayers=self.num_octave_layers,
                                           contrastThreshold=self.contrast_threshold,
                                           edgeThreshold=self.edge_threshold,
                                           sigma=self.sigma)

    def get_system_settings(self):
        """
        Get the settings values used by the detector
        :return:
        """
        return {
            'num_features': self.num_features,
            'num_octave_layers': self.num_octave_layers,
            'contrast_threshold': self.contrast_threshold,
            'edge_threshold': self.edge_threshold,
            'sigma': self.sigma
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
