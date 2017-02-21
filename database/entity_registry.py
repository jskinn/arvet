"""
A minimal module for mapping entity classes to their names in a global registry.
Yes, this prevents effective dependency injection and unit testing,
but I don't have a good way of injecting this to entity when it is declared
"""
__entities = {}


def register_entity(entity_class):
    global __entities
    if hasattr(entity_class, '__name__'):
        __entities[entity_class.__name__] = entity_class


def get_entity_type(entity_type_name):
    global __entities
    if entity_type_name in __entities:
        return __entities[entity_type_name]
