import abc
import database.entity
import core.system


class TrainedVisionSystem(core.system.VisionSystem, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def set_trained_state(self, trained_state):
        """
        Set the trained state of the system, so that it can be run with a particular configuration.
        Should do nothing if the given parameter is not an appropriate state for this system
        :param: TrainedState
        :return:
        """
        pass

    @abc.abstractmethod
    @property
    def trained_state(self):
        """
        Get the id of the current trained state,
        or None if untrained
        :return:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_trainer_class(cls):
        """
        Get the class used for training a particular vision system.
        :return: a subclass of VisionSystemTrainer
        """
        return NotImplemented


class VisionSystemTrainer(metaclass=abc.ABCMeta):
    """
    A separate class like a builder for training Vision Systems.
    This breaks some awkward dependencies with having the system class
    also be responsible for training.
    """

    @abc.abstractmethod
    def set_settings(self, settings):
        """
        Set the setting used when training the vision system.
        The current value of the settings should be baked into
        :param settings:
        :return:
        """
        pass

    @abc.abstractmethod
    def add_dataset(self, dataset_images):
        """
        Add a particular dataset to be trained on.
        This allows the system to be trained on images from multiple datasets.
        :param dataset_images:
        :return:
        """
        pass

    @abc.abstractmethod
    def train_system(self):
        """
        Train the system with the datasets we've already added
        :return: The trial results
        :rtype: TrainedState
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_training_dataset_criteria(cls):
        """
        Get search criteria for finding datasets that can be used by this system for training.
        :return: A set of criteria for finding datasets that match
        :rtype: dict
        """
        return {}


class TrainedState(database.entity.Entity):
    """
    The result of training a particular system wit ha particular dataset
    Contains all the relevant information from the run, and is passed to the benchmark to measure the performance.
    Different subtypes of VisionSystem will have different subclasses of this.

    All Trial results have a one to many relationship with a particular dataset and system.
    """

    def __init__(self, dataset_ids, system_id, system_settings, id_=None, **kwargs):
        super().__init__(id_)
        self._settings = system_settings

        self._dataset_ids = dataset_ids
        self._system_id = system_id

    @property
    def dataset_ids(self):
        """
        The ID of one or more the datasets used to create this result
        :return:
        """
        return self._dataset_ids

    @property
    def system_id(self):
        """
        The ID of the system which produced this result
        :return:
        """
        return self._system_id

    @property
    def settings(self):
        return self._settings

    def consolidate_files(self, new_directory):
        """
        Consolidate any files associated with the trained state into the given folder.
        This is how I manage additional files outside the database.
        Should do nothing if the trained state has no additional files.
        :param new_directory:
        :return:
        """
        pass

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        serialized = super().serialize()
        serialized['datasets'] = self.dataset_ids
        serialized['system'] = self.system_id
        serialized['settings'] = self._settings
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        # massage the compulsory arguments, so that the constructor works
        if 'datasets' in serialized_representation:
            kwargs['dataset_ids'] = serialized_representation['datasets']
        if 'system' in serialized_representation:
            kwargs['system_id'] = serialized_representation['system']
        if 'settings' in serialized_representation:
            kwargs['system_settings'] = serialized_representation['settings']

        return super().deserialize(serialized_representation, **kwargs)
