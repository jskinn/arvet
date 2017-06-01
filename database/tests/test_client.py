import random
import unittest
import unittest.mock as mock
import pymongo
import pymongo.database
import gridfs
import database.client


# Proxy object so that we still have a reference once it has been mocked
MongoClientProxy = pymongo.MongoClient


class TestDatabaseClient(unittest.TestCase):

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_database_name(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        mock_mongoclient.return_value = mock_client_instance

        database_name = 'test_database_name_' + str(random.uniform(-10000, 10000))
        database.client.DatabaseClient({
            'database_config': {
                'database_name': database_name
            }
        })
        self.assertTrue(mock_client_instance.__getitem__.called)
        self.assertIn(mock.call(database_name), mock_client_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_system_trainers_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'system_trainers_collection': collection_name
                }
            }
        })
        _ = db_client.system_trainers_collection  # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_systems_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'system_collection': collection_name
                }
            }
        })
        _ = db_client.system_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_image_source_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'image_source_collection': collection_name
                }
            }
        })
        _ = db_client.image_source_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_image_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'image_collection': collection_name
                }
            }
        })
        _ = db_client.image_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_trials_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'trials_collection': collection_name
                }
            }
        })
        _ = db_client.trials_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_benchmarks_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'benchmarks_collection': collection_name
                }
            }
        })
        _ = db_client.benchmarks_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_results_collection(self, mock_mongoclient, mock_gridfs):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'results_collection': collection_name
                }
            }
        })
        _ = db_client.results_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)
