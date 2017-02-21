class Entity:
    """
    A generic class for all database entities.
    Children of this class can be serialized to the database
    Note that deserialize will call this with some combination of kwargs.
    Either make sure that deserialize only specifies the args,
    or add **kwargs to catch any extras.
    """

    def __init__(self, id_=None, **kwargs):
        self._id = id_

    @property
    def identifier(self):
        """
        Unique id for the entity, for links and references
        :return:
        """
        return self._id

    def refresh_id(self, id_):
        """
        Set the id of the entity, if it didn't have one.
        This is helper for after an entity is saved to the database;
        it's new id is returned, but the entity instance doesn't have it yet, so this method lets us set it.
        As this is the only valid use-case for setting the ID of an entity,
        this function only works if the id is currently None.
        :param id_: The new id of the entity.
        :return: void
        """
        if self._id is None and id_ is not None:
            self._id = id_

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        if self._id is None:
            return {'_type': type(self).__name__}
        else:
            return {'_id': self._id, '_type': type(self).__name__}

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        """
        Deserialize an Entity as retrieved from the database
        Also accepts keyword arguments that are passed directly to the entity constructor,
        so that we can have required parameters to the initialization
        :param serialized_representation: dict
        :return: An instance of the entity class, constructed from the serialized representation
        """
        if '_id' in serialized_representation:
            kwargs['id_'] = serialized_representation['_id']
        entity = cls(**kwargs)
        return entity
