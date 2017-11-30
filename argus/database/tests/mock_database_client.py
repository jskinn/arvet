import unittest.mock as mock
import mongomock
import bson
import gridfs
import argus.database.client


class ZombieDatabaseClient:
    """
    A wrapper around a mock database client that makes it seem to work.
    Much of the heavy lifting is done by the zombie collections, above
    """

    def __init__(self, mock_db_client):
        # Create mock mongo collections for each
        mongomock_client = mongomock.MongoClient()
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
            setattr(mock_db_client, coll_name, mock.Mock(wraps=mongomock_client.db.collection))

        # Model the data storage for gridfs
        self._gridfs_data = {}
        mock_db_client.grid_fs = mock.create_autospec(gridfs.GridFS)
        mock_db_client.grid_fs.get.side_effect = lambda id_: MockReadable(self._gridfs_data[id_])\
            if id_ in self._gridfs_data else None
        mock_db_client.grid_fs.put.side_effect = lambda bytes_: self.put(bytes_)

        # Actually call through for deserialize entity
        mock_db_client.deserialize_entity.side_effect = (
            lambda s_entity, **kwargs:
                argus.database.client.DatabaseClient.deserialize_entity(mock_db_client, s_entity, **kwargs))
        self._mock_db_client = mock_db_client

    @property
    def mock(self) -> argus.database.client.DatabaseClient:
        return self._mock_db_client

    def put(self, bytes_) -> bson.ObjectId:
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


def create() -> ZombieDatabaseClient:
    return ZombieDatabaseClient(mock.create_autospec(argus.database.client.DatabaseClient))
