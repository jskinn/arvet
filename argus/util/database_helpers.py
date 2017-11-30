# Copyright (c) 2017, John Skinner
import copy
import typing
import bson
import pymongo.collection
import argus.database.entity
import argus.database.client


def load_object(db_client: argus.database.client.DatabaseClient, collection: pymongo.collection.Collection,
                id_: bson.ObjectId, **kwargs) -> typing.Union[None, argus.database.entity.Entity]:
    """
    Shorthand helper for pulling a single entity from the database.
    This just saves us creating temporaries for serialized objects all the time,
    and wraps up a bunch of little checks so we don't have to repeat them.

    :param db_client: The database client, for deserializing the entity
    :param collection: The collection to load from
    :param id_: The id of the object to load
    :param kwargs: Additional keword arguments passed to deserialize, optional.
    :return: The deserialized object, or None if it doesn't exist
    """
    if db_client is None or collection is None or id_ is None:
        return None
    s_object = collection.find_one({'_id': id_})
    if s_object is not None:
        return db_client.deserialize_entity(s_object, **kwargs)
    return None


def query_to_dot_notation(query: dict, flatten_arrays: bool = False) -> dict:
    """
    Recursively transform a query containing nested dicts to mongodb dot notation.
    That is,
    {'test': {'a': 1, 'b': 2}} becomes {'test.a': 1, 'test.b': 2}

    Note that this modifies the parameter, and returns it.
    you can either call this as query_to_dot_notation(query) or query = query_to_dot_notation({})

    :param query: The base query as a dict
    :param flatten_arrays: Should arrays be flattened as well. Defaults to false
    :return: query.
    """
    initial_keys = list(query.keys())
    for key in initial_keys:
        if isinstance(query[key], dict):
            query_to_dot_notation(query[key], flatten_arrays=flatten_arrays)
            for inner_key, inner_value in query[key].items():
                query[key+'.'+inner_key] = inner_value
            del query[key]
        elif flatten_arrays and (isinstance(query[key], list) or isinstance(query[key], tuple)):
            for idx, elem in enumerate(query[key]):
                if isinstance(elem, dict):
                    query_to_dot_notation(elem, flatten_arrays=flatten_arrays)
                    for inner_key, inner_value in elem.items():
                        query[key+'.'+str(idx)+'.'+inner_key] = inner_value
                else:
                    query[key+'.'+str(idx)] = elem
            del query[key]
    return query


def add_unique(collection: pymongo.collection.Collection, entity: argus.database.entity.Entity) -> bson.ObjectId:
    """
    Add an object to a collection, if that object does not already exist.
    Treats the entire serialized object as the key, if only one entry is different, they're different objects.
    This ONLY works for very simple entities, more complex objects like image collections
    or image entities have their own save methods that check uniqueness.
    :param collection: The mongodb collection to insert into
    :param entity: The object to insert
    :return: The id of the entity, whether newly added or existing
    """
    if isinstance(entity, dict):
        s_object = entity
    else:
        s_object = entity.serialize()
    query = query_to_dot_notation(copy.deepcopy(s_object))
    existing = collection.find_one(query, {'_id': True})
    if existing is not None:
        return existing['_id']
    else:
        return collection.insert_one(s_object).inserted_id


def add_schema_version(serialized: dict, schema_name: str, version_number: int):
    """
    Add a schema version to a serialized representation. This lets us handle patching and updating dynamically.
    All schemas have a unique name, and a given document may have multiple schemas due to it's inheritance hierarchy.
    For instance, A robot vision system will have schemas for base argus.database.entity.Entity, argus.core.system.VisionSystem,
    and then a third schema for data specific to that system.
    Using this method gives us a standardized way of storing schema versions in the serialized document,
    and helps prevent collisions between the version numbers for different partial schema on the same document.
    :param serialized: The base serialized document, which will have a version number added to it
    :param schema_name: The unique name of the schema to version.
    :param version_number: The version number to set
    :return: void
    """
    if '_schema_version' not in serialized:
        serialized['_schema_version'] = {}
    serialized['_schema_version'][schema_name] = version_number


def get_schema_version(serialized: dict, schema_name: str) -> int:
    """
    Get the version of a particular schema on this document. Defaults to 0 if the version number is missing.
    A document may have multiple schema applied to it by it's inheritance hierarchy, this lets us separate
    the version numbers for each level, so that we can avoid collisions.
    :param serialized: The serialized document
    :param schema_name: The name of the schema to get.
    :return: An integer version number
    """
    if '_schema_version' not in serialized or schema_name not in serialized['_schema_version']:
        return 0
    return serialized['_schema_version'][schema_name]


def check_reference_is_valid(collection: pymongo.collection.Collection, id_: bson.ObjectId) -> bool:
    """
    Check if a given id exists within the given collection
    :param collection: The pymongo collection to search
    :param id_: The id to find
    :return: True if the id exists within the collection, false otherwise
    """
    return collection.find({'_id': id_}).count() > 0
