# Copyright (c) 2017, John Skinner
"""
A minimal module for mapping entity classes to their names in a global registry.
Yes, this prevents effective dependency injection and unit testing,
but I don't have a good way of injecting this to entity when it is declared
"""
import typing


__entities = {}


def get_type_name(entity_class) -> str:
    """
    Get the entity type as it is used in the registry and stored in the database.
    This is useful for creating database queries for entities of a particular type.
    :param entity_class:
    :return:
    """
    return entity_class.__module__ + '.' + entity_class.__name__


def register_entity(entity_class):
    """
    Register an class as an entity.
    The class will be able to be retrieved using the fully-qualified class name
    using get_entity_type below.
    :param entity_class: A class object
    :return: void
    """
    global __entities
    if hasattr(entity_class, '__name__') and hasattr(entity_class, '__module__'):
        __entities[get_type_name(entity_class)] = entity_class


def get_entity_type(entity_type_name: str):
    """
    Get the entity type from its string name.
    If the entity cannot be found, it assumes that the type was not fully qualified,
    and calls find_potential_entity_classes with the type name.
    If in turn this finds exactly one matching class, return that.
    Otherwise, returns None.
    :param entity_type_name: The fully-qualified type name of the class, as a string.
    :return: The entity class object, or None if none was registered
    """
    global __entities
    if entity_type_name in __entities:
        return __entities[entity_type_name]
    else:
        potential_matches = find_potential_entity_classes(entity_type_name)
        if len(potential_matches) == 1:
            return __entities[potential_matches[0]]
    return None


def find_potential_entity_classes(entity_class_name: str) -> typing.List[str]:
    """
    Find fully-qualified entity classes based on just the class name.
    :param entity_class_name:
    :return:
    """
    global __entities
    return [key for key in __entities.keys() if key.rpartition('.')[2] == entity_class_name]
