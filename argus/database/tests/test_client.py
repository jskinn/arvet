# Copyright (c) 2017, John Skinner
import os
import random
import unittest
import unittest.mock as mock
import pymongo
import pymongo.database
import gridfs
import importlib
import argus.database.client


class TestDatabaseClient(unittest.TestCase):

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_temporary_folder(self, *_):
        folder_name = 'arealfolder/anotherrealfolder_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'temp_folder': folder_name
            }
        })
        self.assertEqual(folder_name, db_client.temp_folder)

    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    def test_creates_temporary_folder(self, mock_makedirs, *_):
        folder_name = 'arealfolder/anotherrealfolder_' + str(random.uniform(-10000, 10000))
        argus.database.client.DatabaseClient({
            'database_config': {
                'temp_folder': folder_name
            }
        })
        self.assertTrue(mock_makedirs.called)
        self.assertEqual(folder_name, mock_makedirs.call_args[0][0])

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_database_name(self, mock_mongoclient, *_):
        mock_client_instance = mock_mongoclient.return_value

        database_name = 'test_database_name_' + str(random.uniform(-10000, 10000))
        argus.database.client.DatabaseClient({
            'database_config': {
                'database_name': database_name
            }
        })
        self.assertTrue(mock_client_instance.__getitem__.called)
        self.assertIn(mock.call(database_name), mock_client_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_trainers_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'trainer_collection': collection_name
                }
            }
        })
        _ = db_client.trainer_collection  # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_trainees_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'trainee_collection': collection_name
                }
            }
        })
        _ = db_client.trainee_collection  # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_systems_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'system_collection': collection_name
                }
            }
        })
        _ = db_client.system_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_image_source_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'image_source_collection': collection_name
                }
            }
        })
        _ = db_client.image_source_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_image_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'image_collection': collection_name
                }
            }
        })
        _ = db_client.image_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_trials_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'trials_collection': collection_name
                }
            }
        })
        _ = db_client.trials_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_benchmarks_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'benchmarks_collection': collection_name
                }
            }
        })
        _ = db_client.benchmarks_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_results_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'results_collection': collection_name
                }
            }
        })
        _ = db_client.results_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_experiments_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'experiments_collection': collection_name
                }
            }
        })
        _ = db_client.experiments_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_can_configure_tasks_collection(self, mock_mongoclient, *_):
        database_instance = mock.create_autospec(pymongo.database.Database)
        mock_mongoclient.return_value.__getitem__.return_value = database_instance

        collection_name = 'test_collection_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'collections': {
                    'tasks_collection': collection_name
                }
            }
        })
        _ = db_client.tasks_collection     # Collection is retrieved lazily, so we actually have to ask for it
        self.assertTrue(database_instance.__getitem__.called)
        self.assertIn(mock.call(collection_name), database_instance.__getitem__.call_args_list)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    @mock.patch('argus.database.client.importlib', autospec=importlib)
    def test_deserialize_entity_tries_to_import_module(self, mock_importlib, *_):
        database_name = 'test_database_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'database_name': database_name
            }
        })
        with self.assertRaises(ValueError):
            db_client.deserialize_entity({'_type': 'notamodule.NotAnEntity', 'a': 1})
        self.assertTrue(mock_importlib.import_module.called)
        self.assertEqual(mock.call('notamodule'), mock_importlib.import_module.call_args)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    @mock.patch('argus.database.client.importlib', autospec=importlib)
    def test_deserialize_entity_doesnt_import_empty_module(self, mock_importlib, *_):
        database_name = 'test_database_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'database_name': database_name
            }
        })
        with self.assertRaises(ValueError):
            db_client.deserialize_entity({'_type': 'NotAnEntity', 'a': 1})
        self.assertFalse(mock_importlib.import_module.called)

    @mock.patch('argus.database.client.os.makedirs', autospec=os.makedirs)
    @mock.patch('argus.database.client.importlib', autospec=importlib)
    @mock.patch('argus.database.client.gridfs.GridFS', autospec=gridfs.GridFS)
    @mock.patch('argus.database.client.pymongo.MongoClient', autospec=pymongo.MongoClient)
    def test_deserialize_entity_raises_exception_for_unrecognized_entity_class(self, *_):
        database_name = 'test_database_name_' + str(random.uniform(-10000, 10000))
        db_client = argus.database.client.DatabaseClient({
            'database_config': {
                'database_name': database_name
            }
        })
        with self.assertRaises(ValueError):
            db_client.deserialize_entity({'_type': 'notamodule.NotAnEntity', 'a': 1})
