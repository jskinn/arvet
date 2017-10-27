import unittest.mock as mock
import typing
import bson
import pymongo.collection
import gridfs
import database.client


class ZombieDatabaseCollection:
    """
    A wrapper around a mock collection that lets it actually store and retrieve objects
    """

    def __init__(self, mock_collection: mock.Mock):
        self._elements = {}

        mock_collection.find_one.side_effect = lambda query, *_, **__: self.find_one(query)
        mock_collection.find.side_effect = lambda query, *_, **__: self.find(query)
        mock_collection.insert.side_effect = lambda s_object, *_, **__: self.insert(s_object)
        self._mock_collection = mock_collection

    @property
    def mock(self) -> mock.Mock:
        """
        Get the mock collection this is wrapping
        :return:
        """
        return self._mock_collection

    def insert(self, s_object: dict) -> bson.ObjectId:
        """
        Store a serialized object in the database
        :param s_object:
        :return:
        """
        if '_id' not in s_object:
            s_object['_id'] = bson.ObjectId()
        self._elements[s_object['_id']] = s_object
        return s_object['_id']

    def find_one(self, query: dict) -> dict:
        found = self.find(query)
        return found[0] if len(found) >= 1 else None

    def find(self, query: dict) -> typing.List[dict]:
        if '_id' in query:
            if query['_id'] in self._elements:
                return [self._elements[query['_id']]]
            return []
        else:
            found = []
            for elem in self._elements.values():
                match = True
                for key, val in query:
                    if key not in elem or elem[key] != val:
                        match = False
                        break
                if match:
                    found.append(elem)
            return found


class ZombieDatabaseClient:
    """
    A wrapper around a mock database client that makes it seem to work.
    Much of the heavy lifting is done by the zombie collections, above
    """

    def __init__(self, mock_db_client):
        # Create zombies for each of the collections in the client
        for coll_name in [
            'trainer_collection',
            'trainee_collection',
            'system_collection',
            'image_source_collection',
            'image_collection',
            'trials_collection',
            'benchmarks_collection',
            'results_collection',
            'experiments_collection',
            'tasks_collection'
        ]:
            zombie_collection = ZombieDatabaseCollection(mock.create_autospec(pymongo.collection.Collection))
            setattr(self, coll_name, zombie_collection)
            setattr(mock_db_client, coll_name, zombie_collection.mock)

        # Model the data storage for gridfs
        self._gridfs_data = {}
        self.grid_fs = mock.create_autospec(gridfs.GridFS)
        self.grid_fs.get.side_effect = lambda id_: MockReadable(self._gridfs_data[id_])\
            if id_ in self._gridfs_data else None
        self.grid_fs.put.side_effect = lambda bytes_: self.put(bytes_)

        # Actually call through for deserialize entity
        mock_db_client.deserialize_entity.side_effect = (
            lambda s_entity, **kwargs:
                database.client.DatabaseClient.deserialize_entity(mock_db_client, s_entity, **kwargs))
        self._mock_db_client = mock_db_client

    @property
    def mock(self) -> mock.Mock:
        return self._mock_db_client

    def put(self, bytes_):
        id_ = bson.ObjectId()
        self._gridfs_data[id_] = bytes_
        return id_


class MockReadable:
    """
    A helper for mock gridfs.get to return, that has a 'read' method as expected.
    """
    def __init__(self, bytes_):
        self.bytes = bytes_

    def read(self):
        return self.bytes


def create():
    return ZombieDatabaseClient(mock.create_autospec(database.client.DatabaseClient))
