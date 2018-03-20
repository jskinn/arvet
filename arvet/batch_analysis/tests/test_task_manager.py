# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import bson
import pymongo.collection
import arvet.database.client
import arvet.database.tests.mock_database_client
import arvet.batch_analysis.task_manager as manager
import arvet.batch_analysis.job_system

import arvet.batch_analysis.task
import arvet.batch_analysis.tasks.import_dataset_task as import_dataset_task
import arvet.batch_analysis.tasks.generate_dataset_task as generate_dataset_task
import arvet.batch_analysis.tasks.train_system_task as train_system_task
import arvet.batch_analysis.tasks.run_system_task as run_system_task
import arvet.batch_analysis.tasks.benchmark_trial_task as benchmark_task
# TODO: Tests for these two as well
# import arvet.batch_analysis.tasks.compare_trials_task as compare_trials_task
# import arvet.batch_analysis.tasks.compare_benchmarks_task as compare_benchmarks_task


class TestTaskManager(unittest.TestCase):

    def test_get_import_dataset_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        additional_args = {'foo': 'bar'}
        subject.get_import_dataset_task(module_name, path, additional_args)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('module_name', query)
        self.assertEqual(module_name, query['module_name'])
        self.assertIn('path', query)
        self.assertEqual(path, query['path'])
        self.assertIn('additional_args.foo', query)
        self.assertEqual('bar', query['additional_args.foo'])

    def test_get_import_dataset_task_returns_deserialized_existing(self):
        s_task = {'_type': 'ImportDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_import_dataset_task('lol no', '/tmp/lolno')
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_import_dataset_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        result = subject.get_import_dataset_task(module_name, path)
        self.assertIsInstance(result, import_dataset_task.ImportDatasetTask)
        self.assertIsNone(result.identifier)

    def test_get_generate_dataset_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        for key, value in simulator_config.items():
            self.assertIn('simulator_config.{}'.format(key), query)
            self.assertEqual(value, query['simulator_config.{}'.format(key)])
        self.assertIn('repeat', query)
        self.assertEqual(repeat, query['repeat'])

    def test_get_generate_dataset_task_returns_deserialized_existing(self):
        s_task = {'_type': 'GenerateDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_generate_dataset_task(bson.ObjectId(), bson.ObjectId(), {'provide_rgb': True})
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_generate_dataset_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        result = subject.get_generate_dataset_task(bson.ObjectId(), bson.ObjectId(), {'provide_rgb': True})
        self.assertIsInstance(result, generate_dataset_task.GenerateDatasetTask)
        self.assertIsNone(result.identifier)

    def test_get_train_system_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_train_system_task(bson.ObjectId(), bson.ObjectId())
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_train_system_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trainer_id = bson.ObjectId()
        trainee_id = bson.ObjectId()
        result = subject.get_train_system_task(trainer_id, trainee_id)
        self.assertIsInstance(result, train_system_task.TrainSystemTask)
        self.assertIsNone(result.identifier)

    def test_get_run_system_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_run_system_task(bson.ObjectId(), bson.ObjectId())
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_run_system_task_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        result = subject.get_run_system_task(system_id, image_source_id)
        self.assertIsInstance(result, run_system_task.RunSystemTask)
        self.assertIsNone(result.identifier)

    def test_get_benchmark_task_checks_for_existing_task(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trial_result_ids = [bson.ObjectId(), bson.ObjectId(), bson.ObjectId()]
        benchmark_id = bson.ObjectId()
        subject.get_benchmark_task(trial_result_ids, benchmark_id)

        self.assertTrue(mock_collection.find_one.called)
        query = mock_collection.find_one.call_args[0][0]
        self.assertIn('trial_result_ids', query)
        self.assertIn('$all', query['trial_result_ids'])
        self.assertEqual(set(trial_result_ids), set(query['trial_result_ids']['$all']))
        self.assertIn('benchmark_id', query)
        self.assertEqual(benchmark_id, query['benchmark_id'])

    def test_get_benchmark_task_returns_deserialized_existing(self):
        s_task = {'_type': 'ImportDatasetTask', '_id': bson.ObjectId()}
        mock_entity = mock.MagicMock()
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = s_task
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        mock_db_client.deserialize_entity.return_value = mock_entity
        subject = manager.TaskManager(mock_collection, mock_db_client)

        result = subject.get_benchmark_task([bson.ObjectId()], bson.ObjectId())
        self.assertTrue(mock_db_client.deserialize_entity.called)
        self.assertEqual(s_task, mock_db_client.deserialize_entity.call_args[0][0])
        self.assertEqual(mock_entity, result)

    def test_get_benchmark_returns_new_instance_if_no_existing(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find_one.return_value = None
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trial_result_ids = [bson.ObjectId(), bson.ObjectId(), bson.ObjectId()]
        benchmark_id = bson.ObjectId()
        result = subject.get_benchmark_task(trial_result_ids, benchmark_id)
        self.assertIsInstance(result, benchmark_task.BenchmarkTrialTask)
        self.assertIsNone(result.identifier)

    def test_do_task_checks_import_dataset_task_is_unique(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        repeat = 13
        task = run_system_task.RunSystemTask(system_id, image_source_id, repeat)
        subject.do_task(task)

        self.assertTrue(mock_collection.find.called)
        query = mock_collection.find.call_args[0][0]
        self.assertIn('system_id', query)
        self.assertEqual(system_id, query['system_id'])
        self.assertIn('image_source_id', query)
        self.assertEqual(image_source_id, query['image_source_id'])
        self.assertIn('repeat', query)
        self.assertEqual(repeat, query['repeat'])

    def test_do_task_checks_benchmark_task_is_unique(self):
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        trial_result_ids = [bson.ObjectId(), bson.ObjectId(), bson.ObjectId()]
        benchmark_id = bson.ObjectId()
        task = benchmark_task.BenchmarkTrialTask(trial_result_ids, benchmark_id)
        subject.do_task(task)

        self.assertTrue(mock_collection.find.called)
        query = mock_collection.find.call_args[0][0]
        self.assertIn('trial_result_ids', query)
        self.assertIn('$all', query['trial_result_ids'])
        self.assertEqual(set(trial_result_ids), set(query['trial_result_ids']['$all']))
        self.assertIn('benchmark_id', query)
        self.assertEqual(benchmark_id, query['benchmark_id'])

    def test_do_task_saves_new_task(self):
        # Mock the method chain on the pymongo cursor
        mock_cursor = mock.MagicMock()
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.count.return_value = 0
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_collection.find.return_value = mock_cursor

        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        subject = manager.TaskManager(mock_collection, mock_db_client)
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        task = run_system_task.RunSystemTask(system_id, image_source_id)
        subject.do_task(task)

        self.assertTrue(mock_collection.insert_one.called)
        s_task = task.serialize()
        del s_task['_id']   # This gets set after the insert call, clear it again
        self.assertEqual(s_task, mock_collection.insert_one.call_args[0][0])

    def test_schedule_tasks_cancels_tasks_listed_as_running_on_this_node(self):
        # Set up initial database state
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        this_node_task_id = mock_db_client.tasks_collection.insert_one(import_dataset_task.ImportDatasetTask(
            module_name='test',
            path='/dev/null',
            state=arvet.batch_analysis.task.JobState.RUNNING,
            node_id='here',
            job_id=10
        ).serialize()).inserted_id
        different_node_task_id = mock_db_client.tasks_collection.insert_one(import_dataset_task.ImportDatasetTask(
            module_name='test',
            path='/dev/null',
            state=arvet.batch_analysis.task.JobState.RUNNING,
            node_id='there',
            job_id=10
        ).serialize()).inserted_id

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client, config={
            'task_config': {
                'allow_generate_dataset': False,
                'allow_import_dataset': False,
                'allow_train_system': False,
                'allow_run_system': False,
                'allow_benchmark': False,
                'allow_trial_comparison': False,
                'allow_benchmark_comparison': False
            }
        })

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False

        subject.schedule_tasks(mock_job_system)

        s_task = mock_db_client.tasks_collection.find_one({'_id': this_node_task_id}, {'state': True})
        self.assertEqual(arvet.batch_analysis.task.JobState.UNSTARTED.value, s_task['state'])

        s_task = mock_db_client.tasks_collection.find_one({'_id': different_node_task_id}, {'state': True})
        self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])

    def test_schedule_tasks_schedules_import_dataset_task(self):
        task_entity = import_dataset_task.ImportDatasetTask(
            module_name='test',
            path='/dev/null'
        )
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        task_id = mock_db_client.tasks_collection.insert_one(task_entity.serialize()).inserted_id

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client, config={
            'task_config': {'allow_import_dataset': True}
        })

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        subject.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(
            task_id=task_id,
            num_cpus=task_entity.num_cpus,
            num_gpus=task_entity.num_gpus,
            memory_requirements=task_entity.memory_requirements,
            expected_duration=task_entity.expected_duration
        ), mock_job_system.run_task.call_args)
        s_task = mock_db_client.tasks_collection.find_one({'_id': task_id},
                                                          {'state': True, 'job_id': True, 'node_id': True})
        self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])
        self.assertEqual('here', s_task['node_id'])
        self.assertEqual(1433, s_task['job_id'])

    def test_schedule_tasks_schedules_generate_dataset_task(self):
        task_entity = generate_dataset_task.GenerateDatasetTask(
            controller_id=bson.ObjectId(),
            simulator_id=bson.ObjectId(),
            simulator_config={'foo': 'bar', 'baz': 1},
            repeat=2
        )
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        task_id = mock_db_client.tasks_collection.insert_one(task_entity.serialize()).inserted_id

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client, config={
            'task_config': {'allow_generate_dataset': True}
        })

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        subject.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(
            task_id=task_id,
            num_cpus=task_entity.num_cpus,
            num_gpus=task_entity.num_gpus,
            memory_requirements=task_entity.memory_requirements,
            expected_duration=task_entity.expected_duration
        ), mock_job_system.run_task.call_args)
        s_task = mock_db_client.tasks_collection.find_one({'_id': task_id},
                                                          {'state': True, 'job_id': True, 'node_id': True})
        self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])
        self.assertEqual('here', s_task['node_id'])
        self.assertEqual(1433, s_task['job_id'])

    def test_schedule_tasks_schedules_train_system_task(self):
        task_entity = train_system_task.TrainSystemTask(
            trainer_id=bson.ObjectId(),
            trainee_id=bson.ObjectId()
        )
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        task_id = mock_db_client.tasks_collection.insert_one(task_entity.serialize()).inserted_id

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client, config={
            'task_config': {'allow_train_system': True}
        })

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        subject.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(
            task_id=task_id,
            num_cpus=task_entity.num_cpus,
            num_gpus=task_entity.num_gpus,
            memory_requirements=task_entity.memory_requirements,
            expected_duration=task_entity.expected_duration
        ), mock_job_system.run_task.call_args)
        s_task = mock_db_client.tasks_collection.find_one({'_id': task_id},
                                                          {'state': True, 'job_id': True, 'node_id': True})
        self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])
        self.assertEqual('here', s_task['node_id'])
        self.assertEqual(1433, s_task['job_id'])

    def test_schedule_tasks_schedules_benchmark_trial_task(self):
        task_entity = benchmark_task.BenchmarkTrialTask(
            trial_result_ids=[bson.ObjectId(), bson.ObjectId(), bson.ObjectId()],
            benchmark_id=bson.ObjectId()
        )
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        task_id = mock_db_client.tasks_collection.insert_one(task_entity.serialize()).inserted_id

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client, config={
            'task_config': {'allow_benchmark': True}
        })

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        subject.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(
            task_id=task_id,
            num_cpus=task_entity.num_cpus,
            num_gpus=task_entity.num_gpus,
            memory_requirements=task_entity.memory_requirements,
            expected_duration=task_entity.expected_duration
        ), mock_job_system.run_task.call_args)
        s_task = mock_db_client.tasks_collection.find_one({'_id': task_id},
                                                          {'state': True, 'job_id': True, 'node_id': True})
        self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])
        self.assertEqual('here', s_task['node_id'])
        self.assertEqual(1433, s_task['job_id'])

    def test_schedule_tasks_groups_run_system_tasks_into_batch(self):
        image_source_id_1 = bson.ObjectId()
        image_source_id_2 = bson.ObjectId()
        mock_db_client = arvet.database.tests.mock_database_client.create().mock

        # Create 2 groups of run system tasks, for 2 different image sources
        group_1_ids = []
        for idx in range(10):
            group_1_ids.append(mock_db_client.tasks_collection.insert_one(run_system_task.RunSystemTask(
                system_id=bson.ObjectId(),
                image_source_id=image_source_id_1,
                repeat=idx
            ).serialize()).inserted_id)
        group_2_ids = []
        for idx in range(10):
            group_2_ids.append(mock_db_client.tasks_collection.insert_one(run_system_task.RunSystemTask(
                system_id=bson.ObjectId(),
                image_source_id=image_source_id_2,
                repeat=idx
            ).serialize()).inserted_id)

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client, config={
            'task_config': {'allow_run_system': True}
        })

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        job_ids = [2241, 183]
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433
        mock_job_system.run_script.side_effect = job_ids

        subject.schedule_tasks(mock_job_system)

        self.assertFalse(mock_job_system.run_task.called)
        self.assertEqual(2, len(mock_job_system.run_script.call_args_list))

        # Check that each group of image sources is provided as an argument to a call of run_script
        for image_source_id, task_ids in [
            (image_source_id_1, group_1_ids),
            (image_source_id_2, group_2_ids)
        ]:
            # Find which of the calls to run_script was for this group of tasks
            call = None
            job_id = None
            for idx, temp in enumerate(mock_job_system.run_script.call_args_list):
                if temp[1]['script_args'][1] == str(image_source_id):
                    call = temp
                    job_id = job_ids[idx]
                    break
            self.assertIsNotNone(call)
            self.assertEqual(2 + len(task_ids), len(call[1]['script_args']))
            for task_id in task_ids:
                self.assertIn(str(task_id), call[1]['script_args'])

            # Check all the tasks are now running
            for task_id in task_ids:
                s_task = mock_db_client.tasks_collection.find_one({'_id': task_id},
                                                                  {'state': True, 'job_id': True, 'node_id': True})
                self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])
                self.assertEqual('here', s_task['node_id'])
                self.assertEqual(job_id, s_task['job_id'])

    def test_schedule_dependent_tasks_schedules_tasks(self):
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        task_entities = []
        task_ids = []
        for entity_type, kwargs in [
            (import_dataset_task.ImportDatasetTask, {'module_name': 'test', 'path': '/dev/null'}),
            (generate_dataset_task.GenerateDatasetTask, {
                'controller_id': bson.ObjectId(),
                'simulator_id': bson.ObjectId(),
                'simulator_config': {'foo': 'bar', 'baz': 1},
                'repeat': 2}),
            (train_system_task.TrainSystemTask, {'trainer_id': bson.ObjectId(), 'trainee_id': bson.ObjectId()}),
            (run_system_task.RunSystemTask, {'system_id': bson.ObjectId(), 'image_source_id': bson.ObjectId(),
                                             'repeat': 12}),
            (benchmark_task.BenchmarkTrialTask, {
                'trial_result_ids': [bson.ObjectId(), bson.ObjectId()], 'benchmark_id': bson.ObjectId()
            })
        ]:
            task_entity = entity_type(
                job_id=823,
                node_id='there',
                state=arvet.batch_analysis.task.JobState.RUNNING, **kwargs)
            task_entities.append(task_entity)
            task_ids.append(mock_db_client.tasks_collection.insert_one(task_entity.serialize()).inserted_id)

        subject = manager.TaskManager(mock_db_client.tasks_collection, mock_db_client)

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        subject.schedule_dependent_tasks(task_ids, mock_job_system)

        for idx in range(len(task_ids)):
            self.assertIn(mock.call(
                task_id=task_ids[idx],
                num_cpus=task_entities[idx].num_cpus,
                num_gpus=task_entities[idx].num_gpus,
                memory_requirements=task_entities[idx].memory_requirements,
                expected_duration=task_entities[idx].expected_duration
            ), mock_job_system.run_task.call_args_list)
            s_task = mock_db_client.tasks_collection.find_one({'_id': task_ids[idx]},
                                                              {'state': True, 'job_id': True, 'node_id': True})
            self.assertEqual(arvet.batch_analysis.task.JobState.RUNNING.value, s_task['state'])
            self.assertEqual('here', s_task['node_id'])
            self.assertEqual(1433, s_task['job_id'])
