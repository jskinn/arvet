# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import pymongo
import bson
import core.trial_result
import core.sequence_type
import core.tests.mock_types as mock_core
import batch_analysis.tests.mock_task_manager as mock_manager_factory
import database.tests.mock_database_client as mock_client_factory
import database.tests.test_entity
import database.client
import util.dict_utils as du
import batch_analysis.experiment as ex


# A minimal experiment, so we can instantiate the abstract class
class MockExperiment(ex.Experiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_imports(self, task_manager, db_client):
        pass

    def schedule_tasks(self, task_manager, db_client):
        pass


class TestExperiment(unittest.TestCase, database.tests.test_entity.EntityContract):

    def get_class(self):
        return MockExperiment

    def make_instance(self, *args, **kwargs):
        du.defaults(kwargs, {
            'trial_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId}},
            'result_map': {bson.ObjectId(): {bson.ObjectId(): bson.ObjectId}}
        })
        return MockExperiment(*args, **kwargs)

    def assert_models_equal(self, task1, task2):
        """
        Helper to assert that two tasks are equal
        We're going to violate encapsulation for a bit
        :param task1:
        :param task2:
        :return:
        """
        if (not isinstance(task1, ex.Experiment) or
                not isinstance(task2, ex.Experiment)):
            self.fail('object was not an Experiment')
        self.assertEqual(task1.identifier, task2.identifier)
        self.assertEqual(task1._trial_map, task2._trial_map)
        self.assertEqual(task1._result_map, task2._result_map)

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

    def test_schedule_all_schedules_all_trial_combinations(self):
        zombie_db_client = mock_client_factory.create()
        zombie_task_manager = mock_manager_factory.create()
        subject = MockExperiment()
        systems = []
        image_sources = []
        for _ in range(3):
            entity = mock_core.MockSystem()
            systems.append(zombie_db_client.system_collection.insert(entity.serialize()))
            entity = mock_core.MockImageSource()
            image_sources.append(zombie_db_client.image_source_collection.insert(entity.serialize()))
        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, [])

        for system_id in systems:
            for image_source_id in image_sources:
                self.assertIn(mock.call(system_id=system_id, image_source_id=image_source_id,
                                        memory_requirements=mock.ANY, expected_duration=mock.ANY),
                              zombie_task_manager.mock.get_run_system_task.call_args_list)

    def test_schedule_all_schedules_all_benchmark_combinations(self):
        zombie_db_client = mock_client_factory.create()
        zombie_task_manager = mock_manager_factory.create()
        subject = MockExperiment()
        systems = []
        image_sources = []
        trial_results = []
        benchmarks = []
        for _ in range(3):  # Create
            entity = mock_core.MockSystem()
            systems.append(zombie_db_client.system_collection.insert(entity.serialize()))

            entity = mock_core.MockImageSource()
            image_sources.append(zombie_db_client.image_source_collection.insert(entity.serialize()))

            entity = mock_core.MockBenchmark()
            benchmarks.append(zombie_db_client.benchmarks_collection.insert(entity.serialize()))

        for system_id in systems:
            for image_source_id in image_sources:
                # Create trial results for each combination of systems and image sources,
                # as if the run system tasks are complete
                entity = core.trial_result.TrialResult(
                    system_id, True, core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                trial_result_id = zombie_db_client.trials_collection.insert(entity.serialize())
                trial_results.append(trial_result_id)
                task = zombie_task_manager.get_run_system_task(system_id, image_source_id)
                task.mark_job_started('test', 0)
                task.mark_job_complete(trial_result_id)

        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, benchmarks)

        # Check if we scheduled all the benchmarks
        for trial_result_id in trial_results:
            for benchmark_id in benchmarks:
                self.assertIn(mock.call(trial_result_id=trial_result_id, benchmark_id=benchmark_id,
                                        memory_requirements=mock.ANY, expected_duration=mock.ANY),
                              zombie_task_manager.mock.get_benchmark_task.call_args_list)
