#Copyright (c) 2017, John Skinner
import abc
import database.identifiable as identifiable
import database.entity_registry as reg


class EntityMetaclass(type):
    """
    Entity metaclass.
    When an entity class is declared, it will register the class name in the entity register.
    This is so that type names stored as strings in the database can be turned back to entity instances.
    """
    def __init__(cls, name, bases, dict_):
        super().__init__(name, bases, dict_)
        reg.register_entity(cls)


class AbstractEntityMetaclass(EntityMetaclass, abc.ABCMeta):
    """
    A simple combination class for entity classes that are also abstract classes.
    """
    def __init__(cls, name, bases, dict_):
        super().__init__(name, bases, dict_)


class Entity(identifiable.Identifiable, metaclass=AbstractEntityMetaclass):
    """
    A generic class for all database entities.
    Children of this class can be serialized to the database
    Note that deserialize will call this with some combination of kwargs.
    Either make sure that deserialize only specifies the args,
    or add **kwargs to catch any extras.
    """

    def __init__(self, id_=None, **kwargs):
        self._id = id_
        super().__init__(**kwargs)

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

    def validate(self):
        """
        Check that the entity is valid.
        This lets us check things like whether references files still exist.
        :return: True iff the entity is in a valid state, and can be used.
        """
        return True

    def serialize(self):
        """
        Serialize the entity to a dict format, that can be saved to the database
        :return: a dictionary representation of the entity, that can be saved to MongoDB
        """
        type_ = type(self)
        if self._id is None:
            return {'_type': type_.__module__ + '.' + type_.__name__}
        else:
            return {'_id': self._id, '_type': type_.__module__ + '.' + type_.__name__}

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        """
        Deserialize an Entity as retrieved from the database
        Also accepts keyword arguments that are passed directly to the entity constructor,
        so that we can have required parameters to the initialization
        :param serialized_representation: dict
        :param db_client: The database client, which is used to deserialize linked entities
        :return: An instance of the entity class, constructed from the serialized representation
        """
        if '_id' in serialized_representation:
            kwargs['id_'] = serialized_representation['_id']
        entity = cls(**kwargs)
        return entity
