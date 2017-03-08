import database.entity


class TrialResult(database.entity.Entity):
    """
    The result of running a particular system with images from a particular image source.
    Contains all the relevant information from the run, and is passed to the benchmark to measure the performance.
    Different subtypes of VisionSystem will have different subclasses of this.

    All Trial results have a one to many relationship with a particular dataset and system.
    """

    def __init__(self, image_source_id, system_id, success, system_settings, id_=None, **kwargs):
        super().__init__(id_, **kwargs)
        self._success = bool(success)
        self._settings = system_settings

        self._image_source_id = image_source_id
        self._system_id = system_id

    @property
    def image_source_id(self):
        """
        The ID of the image source used when performing this trial
        :return:
        """
        return self._image_source_id

    @property
    def system_id(self):
        """
        The ID of the system which produced this result
        :return:
        """
        return self._system_id

    @property
    def success(self):
        """
        Did the run succeed or not.
        A system may crash, or fail in some way on a particular dataset,
        which should produce a TrialResult with success as False

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
        serialized['image_source'] = self.image_source_id
        serialized['system'] = self.system_id
        serialized['success'] = self.success
        serialized['settings'] = self._settings
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        """
        Deserialize a Trial Result
        :param serialized_representation:
        :param kwargs:
        :return:
        """
        # massage the compulsory arguments, so that the constructor works
        if 'image_source' in serialized_representation:
            kwargs['image_source_id'] = serialized_representation['image_source']
        if 'system' in serialized_representation:
            kwargs['system_id'] = serialized_representation['system']
        if 'success' in serialized_representation:
            kwargs['success'] = serialized_representation['success']
        if 'settings' in serialized_representation:
            kwargs['system_settings'] = serialized_representation['settings']

        return super().deserialize(serialized_representation, **kwargs)


class FailedTrial(TrialResult):
    """
    A Trial Result that has failed.
    All Trial Results with success == false should be an instance of
    this class or a subclass of this class.
    Think of this like an exception returned from a run system call.
    """
    def __init__(self, image_source_id, system_id, reason, system_settings, id_=None, **kwargs):
        super().__init__(image_source_id, system_id, False, system_settings, id_=id_)
        self._reason = reason

    @property
    def reason(self):
        """
        The reason the trial failed, for diagnosis and debug.
        :return: The string reason passed to the constructor
        :rtype str:
        """
        return self._reason

    def serialize(self):
        """
        Serialize a FailedTrial for storing in a database
        :return: A dict representation of this object.
        """
        serialized = super().serialize()
        serialized['reason'] = self.reason
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        """
        Deserialize a FailedTrial
        :param serialized_representation:
        :param kwargs: keyword arguments to the entity constructor. Are overridden by the serialized representation.
        :return: a FailedTrial object
        """
        # massage the compulsory arguments, so that the constructor works
        if 'reason' in serialized_representation:
            kwargs['reason'] = serialized_representation['reason']
        return super().deserialize(serialized_representation, **kwargs)
