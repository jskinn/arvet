import random
import unittest
import unittest.mock as mock
import pymongo
import pymongo.database
import database.client


# Proxy object so that we still have a reference once it has been mocked
MongoClientProxy = pymongo.MongoClient


class TestDatabaseClient(unittest.TestCase):

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_database_name(self, mock_mongoclient):
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

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_systems_collection(self, mock_mongoclient):
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

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_image_source_collection(self, mock_mongoclient):
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

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_image_collection(self, mock_mongoclient):
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

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_trials_collection(self, mock_mongoclient):
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

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_results_collection(self, mock_mongoclient):
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

    @mock.patch('pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_trained_state_collection(self, mock_mongoclient):
        mock_client_instance = mock.create_autospec(MongoClientProxy)
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = database_instance


        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'trained_state_collection': collection_name
                }
            }
        })
        _ = db_client.trained_state_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)
