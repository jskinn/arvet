#Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import bson
import pymongo.collection
import database.client
import batch_analysis.task_manager as manager

import batch_analysis.tasks.import_dataset_task as import_dataset_task
import batch_analysis.tasks.generate_dataset_task as generate_dataset_task
import batch_analysis.tasks.train_system_task as train_system_task
import batch_analysis.tasks.run_system_task as run_system_task
import batch_analysis.tasks.benchmark_trial_task as benchmark_task
# TODO: Tests for these two as well
# import batch_analysis.tasks.compare_trials_task as compare_trials_task
# import batch_analysis.tasks.compare_benchmarks_task as compare_benchmarks_task


class TestTaskManager(unittest.TestCase):

    def test_get_import_dataset_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        subject.get_import_dataset_task(module_name, path)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('module_name', query)
        self.assertEqual(module_name, query['module_name'])
        self.assertIn('path', query)
        self.assertEqual(path, query['path'])

    def test_get_import_dataset_task_returns_deserialized_existing(self):
        s_task = {'_type': 'ImportDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_import_dataset_task('lol no', '/tmp/lolno')
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_import_dataset_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        result = subject.get_import_dataset_task(module_name, path)
        self.assertIsInstance(result, import_dataset_task.ImportDatasetTask)
        self.assertIsNone(result.identifier)

    def test_get_generate_dataset_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        controller_id = bson.ObjectId()
        simulator_id = bson.ObjectId()
        simulator_config = {
            'stereo_offset': 0.15,
            'provide_rgb': True,
            'provide_depth': True,
            'provide_labels': False,
            'provide_world_normals': False
        }
        repeat = 170
        subject.get_generate_dataset_task(controller_id, simulator_id, simulator_config, repeat)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('controller_id', query)
        self.assertEqual(controller_id, query['controller_id'])
        self.assertIn('simulator_id', query)
        self.assertEqual(simulator_id, query['simulator_id'])
        self.assertIn('simulator_config', query)
        self.assertEqual(simulator_config, query['simulator_config'])
        self.assertIn('repeat', query)
        self.assertEqual(repeat, query['repeat'])

    def test_get_generate_dataset_task_returns_deserialized_existing(self):
        s_task = {'_type': 'GenerateDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_generate_dataset_task(bson.ObjectId(), bson.ObjectId(), {'provide_rgb': True})
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_generate_dataset_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        result = subject.get_generate_dataset_task(bson.ObjectId(), bson.ObjectId(), {'provide_rgb': True})
        self.assertIsInstance(result, generate_dataset_task.GenerateDatasetTask)
        self.assertIsNone(result.identifier)

    def test_get_train_system_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trainer_id = bson.ObjectId()
        trainee_id = bson.ObjectId()
        subject.get_train_system_task(trainer_id, trainee_id)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('trainer_id', query)
        self.assertEqual(trainer_id, query['trainer_id'])
        self.assertIn('trainee_id', query)
        self.assertEqual(trainee_id, query['trainee_id'])

    def test_get_train_system_task_returns_deserialized_existing(self):
        s_task = {'_type': 'ImportDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_train_system_task(bson.ObjectId(), bson.ObjectId())
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_train_system_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trainer_id = bson.ObjectId()
        trainee_id = bson.ObjectId()
        result = subject.get_train_system_task(trainer_id, trainee_id)
        self.assertIsInstance(result, train_system_task.TrainSystemTask)
        self.assertIsNone(result.identifier)

    def test_get_run_system_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        subject.get_run_system_task(system_id, image_source_id)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('system_id', query)
        self.assertEqual(system_id, query['system_id'])
        self.assertIn('image_source_id', query)
        self.assertEqual(image_source_id, query['image_source_id'])

    def test_get_run_system_task_returns_deserialized_existing(self):
        s_task = {'_type': 'ImportDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_run_system_task(bson.ObjectId(), bson.ObjectId())
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_run_system_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        result = subject.get_run_system_task(system_id, image_source_id)
        self.assertIsInstance(result, run_system_task.RunSystemTask)
        self.assertIsNone(result.identifier)

    def test_get_benchmark_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trial_result_id = bson.ObjectId()
        benchmark_id = bson.ObjectId()
        subject.get_benchmark_task(trial_result_id, benchmark_id)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('trial_result_id', query)
        self.assertEqual(trial_result_id, query['trial_result_id'])
        self.assertIn('benchmark_id', query)
        self.assertEqual(benchmark_id, query['benchmark_id'])

    def test_get_benchmark_task_returns_deserialized_existing(self):
        s_task = {'_type': 'ImportDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_benchmark_task(bson.ObjectId(), bson.ObjectId())
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_benchmark_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trial_result_id = bson.ObjectId()
        benchmark_id = bson.ObjectId()
        result = subject.get_benchmark_task(trial_result_id, benchmark_id)
        self.assertIsInstance(result, benchmark_task.BenchmarkTrialTask)
        self.assertIsNone(result.identifier)

    def test_do_task_checks_import_benchmark_task_is_unique(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        task = import_dataset_task.ImportDatasetTask(module_name, path)
        subject.do_task(task)

        self.assertTrue(mock_collection.find.called)
        query = mock_collection.find.call_args[0][0]
        self.assertIn('module_name', query)
        self.assertEqual(module_name, query['module_name'])
        self.assertIn('path', query)
        self.assertEqual(path, query['path'])

    def test_do_task_checks_train_system_task_is_unique(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trainer_id = bson.ObjectId()
        trainee_id = bson.ObjectId()
        task = train_system_task.TrainSystemTask(trainer_id, trainee_id)
        subject.do_task(task)

        self.assertTrue(mock_collection.find.called)
        query = mock_collection.find.call_args[0][0]
        self.assertIn('trainer_id', query)
        self.assertEqual(trainer_id, query['trainer_id'])
        self.assertIn('trainee_id', query)
        self.assertEqual(trainee_id, query['trainee_id'])

    def test_do_task_checks_run_system_task_is_unique(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        task = run_system_task.RunSystemTask(system_id, image_source_id)
        subject.do_task(task)

        self.assertTrue(mock_collection.find.called)
        query = mock_collection.find.call_args[0][0]
        self.assertIn('system_id', query)
        self.assertEqual(system_id, query['system_id'])
        self.assertIn('image_source_id', query)
        self.assertEqual(image_source_id, query['image_source_id'])

    def test_do_task_checks_benchmark_task_is_unique(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trial_result_id = bson.ObjectId()
        benchmark_id = bson.ObjectId()
        task = benchmark_task.BenchmarkTrialTask(trial_result_id, benchmark_id)
        subject.do_task(task)

        self.assertTrue(mock_collection.find.called)
        query = mock_collection.find.call_args[0][0]
        self.assertIn('trial_result_id', query)
        self.assertEqual(trial_result_id, query['trial_result_id'])
        self.assertIn('benchmark_id', query)
        self.assertEqual(benchmark_id, query['benchmark_id'])

    def test_do_task_saves_new_task(self):
        # Mockthe method chain on the pymongo cursor
        mock_cursor = mock.MagicMock()
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.count.return_value = 0
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find.return_value = mock_cursor

        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        task = run_system_task.RunSystemTask(system_id, image_source_id)
        subject.do_task(task)

        self.assertTrue(mock_collection.insert.called)
        s_task = task.serialize()
        del s_task['_id']   # This gets set after the insert call, clear it again
        self.assertEqual(s_task, mock_collection.insert.call_args[0][0])
