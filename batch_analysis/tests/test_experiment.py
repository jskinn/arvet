import unittest
import unittest.mock as mock
import pymongo
import bson
import database.tests.test_entity
import database.client
import batch_analysis.experiment as ex


# A minimal experiment, so we can instantiate the abstract class
class MockExperiment(ex.Experiment):
    def do_imports(self, task_manager, db_client):
        pass

    def schedule_tasks(self, task_manager, db_client):
        pass


class TestExperiment(unittest.TestCase):

    def test_constructor_works_with_minimal_arguments(self):
        MockExperiment()

    def test_save_updates_inserts_if_no_id(self):
        new_id = bson.ObjectId()
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.experiments_collection.insert.return_value = new_id

        test_id = bson.ObjectId()
        test_id2 = bson.ObjectId()
        subject = MockExperiment()
        subject._add_to_set('test', [test_id])
        subject._set_property('test2', test_id2)
        subject.save_updates(mock_db_client)

        self.assertTrue(mock_db_client.experiments_collection.insert.called)
        s_subject = subject.serialize()
        del s_subject['_id']    # This key didn't exist when it was first serialized
        self.assertEqual(s_subject, mock_db_client.experiments_collection.insert.call_args[0][0])
        self.assertEqual(new_id, subject.identifier)

    def test_save_updates_stores_accumulated_changes(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)

        test_id = bson.ObjectId()
        test_id2 = bson.ObjectId()
        subject = MockExperiment(id_=bson.ObjectId())
        subject._add_to_set('test', [test_id])
        subject._set_property('test2', test_id2)
        subject.save_updates(mock_db_client)

        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
            '_id': subject.identifier
        }, {
            '$set': {
                "test2": test_id2
            },
            '$addToSet': {
                'test': {'$each': [test_id]}
            }
        }), mock_db_client.experiments_collection.update.call_args)

    def test_save_updates_does_nothing_if_no_changes(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        subject = MockExperiment(id_=bson.ObjectId())
        subject.save_updates(mock_db_client)
        self.assertFalse(mock_db_client.experiments_collection.update.called)
