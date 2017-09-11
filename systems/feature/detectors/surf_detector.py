import cv2
import cv2.xfeatures2d
import util.dict_utils as du
import systems.feature.feature_detector


class SurfDetector(systems.feature.feature_detector.FeatureDetector):
    """
    A FeatureDetector that uses SURF features
    """

    def __init__(self, config=None, id_=None, **kwargs):
        super().__init__(id_=id_, **kwargs)
        if config is None:
            config = {}
        config = du.defaults({}, config, kwargs, {
            'hessian_threshold': 100,
            'num_octaves': 4,
            'num_octave_layers': 3,
            'extended': False,
            'upright': False
        })

        self._hessian_threshold = float(config['hessian_threshold'])
        self._num_octaves = int(config['num_octaves'])
        self._num_octave_layers = int(config['num_octave_layers'])
        self._extended = bool(config['extended'])
        self._upright = bool(config['upright'])

        # variables for trial state
        self._surf = None
        self._key_points = None

    @property
    def hessian_threshold(self):
        return self._hessian_threshold

    @hessian_threshold.setter
    def hessian_threshold(self, hessian_threshold):
        if not self.is_trial_running():
            self._hessian_threshold = float(hessian_threshold)

    @property
    def num_octaves(self):
        return self._num_octaves

    @num_octaves.setter
    def num_octaves(self, number_of_octaves):
        if not self.is_trial_running():
            self._num_octaves = int(number_of_octaves)

    @property
    def num_octave_layers(self):
        return self._num_octave_layers

    @num_octave_layers.setter
    def num_octave_layers(self, number_of_octave_layers):
        if not self.is_trial_running():
            self._num_octave_layers = int(number_of_octave_layers)

    @property
    def extended(self):
        return self._extended

    @extended.setter
    def extended(self, extended):
        if not self.is_trial_running():
            self._extended = bool(extended)

    @property
    def upright(self):
        return self._upright

    @upright.setter
    def upright(self, upright):
        if not self.is_trial_running():
            self._upright = bool(upright)

    def make_detector(self):
        """
        Make the SIFT detector for this system.
        :return:
        """
        return cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessian_threshold,
                                           nOctaves=self.num_octaves,
                                           nOctaveLayers=self.num_octave_layers,
                                           extended=self.extended,
                                           upright=self.upright)

    def get_system_settings(self):
        """
        Get the settings values used by the detector
        :return:
        """
        return {
            'hessian_threshold': self.hessian_threshold,
            'num_octaves': self.num_octaves,
            'num_octave_layers': self.num_octave_layers,
            'extended': self.extended,
            'upright': self.upright
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
