import database.entity


class TrialResult(database.entity.Entity):
    """
    The result of running a particular system with a particular dataset.
    Contains all the relevant information from the run, and is passed to the benchmark to measure the performance.
    Different subtypes of VisionSystem will have different subclasses of this.

    All Trial results have a one to many relationship with a particular dataset and system.
    """

    def __init__(self, dataset_id, system_id, trained_state_id, success, system_settings, id_=None, **kwargs):
        super().__init__(id_)
        self._success = bool(success)
        self._settings = system_settings

        self._dataset_id = dataset_id
        self._system_id = system_id
        self._trained_state = trained_state_id

    @property
    def dataset_id(self):
        """
        The ID of the dataset used to create this result
        :return:
        """
        return self._dataset_id

    @property
    def system_id(self):
        """
        The ID of the system which produced this result
        :return:
        """
        return self._system_id

    @property
    def trained_state_id(self):
        """
        For a trained system, the id of the trained state used to produce the result.
        None for untrained systems
        :return:
        """
        return self._trained_state

    @property
    def success(self):
        """
        Did the run succeed or not.
        A system may crash, or fail in some way on a particular dataset, which should produce a TrialResult with

        :return: True iff the trial succeeded
        :rtype: bool
        """
        return self._success

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        serialized = super().serialize()
        serialized['dataset'] = self.dataset_id
        serialized['system'] = self.system_id
        serialized['trained_state'] = self.trained_state_id
        serialized['success'] = self.success
        serialized['settings'] = self._settings
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        # massage the compulsory arguments, so that the constructor works
        if 'dataset' in serialized_representation:
            kwargs['dataset_id'] = serialized_representation['dataset']
        if 'system' in serialized_representation:
            kwargs['system_id'] = serialized_representation['system']
        if 'trained_state' in serialized_representation:
            kwargs['trained_state_id'] = serialized_representation['trained_state']
        if 'success' in serialized_representation:
            kwargs['success'] = serialized_representation['success']
        if 'settings' in serialized_representation:
            kwargs['system_settings'] = serialized_representation['settings']

        return super().deserialize(serialized_representation, **kwargs)
