# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import pymongo
import bson
import arvet.core.trial_result
import arvet.core.benchmark
import arvet.core.sequence_type
import arvet.core.tests.mock_types as mock_core
import arvet.batch_analysis.tests.mock_task_manager as mock_manager_factory
import arvet.database.tests.mock_database_client as mock_client_factory
import arvet.database.tests.test_entity
import arvet.database.client
import arvet.util.dict_utils as du
import arvet.batch_analysis.experiment as ex


# A minimal experiment, so we can instantiate the abstract class
class MockExperiment(ex.Experiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_imports(self, task_manager, path_manager, db_client):
        pass

    def schedule_tasks(self, task_manager, db_client):
        pass


class TestExperiment(unittest.TestCase, arvet.database.tests.test_entity.EntityContract):

    def setUp(self):
        # Some complete results, so that we can have a non-empty trial map and result map on entity tests
        self.systems = [mock_core.MockSystem()]
        self.image_sources = [mock_core.MockImageSource() for _ in range(3)]
        self.trial_results = [arvet.core.trial_result.TrialResult(
                    system.identifier, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
            for _ in self.image_sources for system in self.systems]
        self.benchmarks = [mock_core.MockBenchmark() for _ in range(2)]
        self.benchmark_results = [arvet.core.benchmark.BenchmarkResult(benchmark.identifier, trial_result.identifier,
                                                                       True)
                                  for benchmark in self.benchmarks for trial_result in self.trial_results]

    def get_class(self):
        return MockExperiment

    def make_instance(self, *args, **kwargs):
        du.defaults(kwargs, {
            'trial_map': {system.identifier: {image_source.identifier: trial_result.identifier
                                              for trial_result in self.trial_results
                                              if trial_result.identifier is not None and
                                              trial_result.system_id == system.identifier
                                              for image_source in self.image_sources
                                              if image_source.identifier is not None}
                          for system in self.systems if system.identifier is not None},
            'result_map': {trial_result.identifier: {benchmark.identifier: benchmark_result.identifier
                                                     for benchmark in self.benchmarks
                                                     if benchmark.identifier is not None
                                                     for benchmark_result in self.benchmark_results
                                                     if benchmark_result.identifier is not None and
                                                     benchmark_result.benchmark == benchmark.identifier and
                                                     benchmark_result.trial_result == trial_result.identifier}
                           for trial_result in self.trial_results
                           if trial_result.identifier is not None}
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

    def create_mock_db_client(self):
        mock_db_client = super().create_mock_db_client()

        # Insert all the background data we expect.
        for system in self.systems:
            result = mock_db_client.system_collection.insert_one(system.serialize())
            system.refresh_id(result.inserted_id)
        for image_source in self.image_sources:
            result = mock_db_client.image_source_collection.insert_one(image_source.serialize())
            image_source.refresh_id(result.inserted_id)
        for trial_result in self.trial_results:
            result = mock_db_client.trials_collection.insert_one(trial_result.serialize())
            trial_result.refresh_id(result.inserted_id)
        for benchmark in self.benchmarks:
            result = mock_db_client.benchmarks_collection.insert_one(benchmark.serialize())
            benchmark.refresh_id(result.inserted_id)
        for benchmark_result in self.benchmark_results:
            result = mock_db_client.results_collection.insert_one(benchmark_result.serialize())
            benchmark_result.refresh_id(result.inserted_id)

        return mock_db_client

    def test_constructor_works_with_minimal_arguments(self):
        MockExperiment()

    def test_save_updates_inserts_if_no_id(self):
        new_id = bson.ObjectId()
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
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
        mock_db_client = mock.create_autospec(arvet.database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        subject = MockExperiment(id_=bson.ObjectId())
        subject.save_updates(mock_db_client)
        self.assertFalse(mock_db_client.experiments_collection.update.called)

    def test_schedule_all_schedules_all_trial_combinations(self):
        zombie_task_manager = mock_manager_factory.create()
        mock_db_client = self.create_mock_db_client()
        subject = MockExperiment()
        systems = []
        image_sources = []
        for _ in range(3):
            entity = mock_core.MockSystem()
            systems.append(mock_db_client.system_collection.insert_one(entity.serialize()).inserted_id)
            entity = mock_core.MockImageSource()
            image_sources.append(mock_db_client.image_source_collection.insert_one(entity.serialize()).inserted_id)
        subject.schedule_all(zombie_task_manager.mock, mock_db_client, systems, image_sources, [])

        for system_id in systems:
            for image_source_id in image_sources:
                self.assertIn(mock.call(system_id=system_id, image_source_id=image_source_id, repeat=0,
                                        memory_requirements=mock.ANY, expected_duration=mock.ANY),
                              zombie_task_manager.mock.get_run_system_task.call_args_list)

    def test_schedule_all_schedules_repeats(self):
        zombie_task_manager = mock_manager_factory.create()
        mock_db_client = self.create_mock_db_client()
        subject = MockExperiment()
        systems = []
        image_sources = []
        repeats = 5
        for _ in range(3):
            entity = mock_core.MockSystem()
            systems.append(mock_db_client.system_collection.insert_one(entity.serialize()).inserted_id)
            entity = mock_core.MockImageSource()
            image_sources.append(mock_db_client.image_source_collection.insert_one(entity.serialize()).inserted_id)
        subject.schedule_all(zombie_task_manager.mock, mock_db_client, systems, image_sources, [], repeats=repeats)

        for system_id in systems:
            for image_source_id in image_sources:
                for repeat in range(repeats):
                    self.assertIn(mock.call(system_id=system_id, image_source_id=image_source_id, repeat=repeat,
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
            systems.append(zombie_db_client.mock.system_collection.insert_one(entity.serialize()).inserted_id)

            entity = mock_core.MockImageSource()
            image_sources.append(
                zombie_db_client.mock.image_source_collection.insert_one(entity.serialize()).inserted_id)

            entity = mock_core.MockBenchmark()
            benchmarks.append(zombie_db_client.mock.benchmarks_collection.insert_one(entity.serialize()).inserted_id)

        for system_id in systems:
            for image_source_id in image_sources:
                # Create trial results for each combination of systems and image sources,
                # as if the run system tasks are complete
                entity = arvet.core.trial_result.TrialResult(
                    system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                trial_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
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

    def test_schedule_all_stores_trial_results(self):
        mock_db_client = self.create_mock_db_client()
        zombie_task_manager = mock_manager_factory.create()
        subject = MockExperiment()
        systems = []
        image_sources = []
        expected_trial_results = []
        for _ in range(3):  # Create systems and image sources
            entity = mock_core.MockSystem()
            systems.append(mock_db_client.system_collection.insert_one(entity.serialize()).inserted_id)
            entity = mock_core.MockImageSource()
            image_sources.append(mock_db_client.image_source_collection.insert_one(entity.serialize()).inserted_id)

        for system_id in systems:
            for image_source_id in image_sources:
                # Create trial results for each combination of systems and image sources,
                # as if the run system tasks are complete
                entity = arvet.core.trial_result.TrialResult(
                    system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                trial_result_id = mock_db_client.trials_collection.insert_one(entity.serialize()).inserted_id
                expected_trial_results.append(trial_result_id)
                task = zombie_task_manager.get_run_system_task(system_id, image_source_id)
                task.mark_job_started('test', 0)
                task.mark_job_complete(trial_result_id)

        subject.schedule_all(zombie_task_manager.mock, mock_db_client, systems, image_sources, [])

        for system_id in systems:
            for image_source_id in image_sources:
                trial_result_ids = subject.get_trial_results(system_id, image_source_id)
                self.assertIsNotNone(trial_result_ids)
                for trial_result_id in trial_result_ids:
                    self.assertIn(trial_result_id, expected_trial_results)

    def test_schedule_all_stores_benchmark_results(self):

        zombie_db_client = mock_client_factory.create()
        zombie_task_manager = mock_manager_factory.create()
        subject = MockExperiment()
        systems = []
        image_sources = []
        trial_results = []
        benchmarks = []
        benchmark_results = []
        for _ in range(3):  # Create
            entity = mock_core.MockSystem()
            systems.append(zombie_db_client.mock.system_collection.insert_one(entity.serialize()).inserted_id)

            entity = mock_core.MockImageSource()
            image_sources.append(
                zombie_db_client.mock.image_source_collection.insert_one(entity.serialize()).inserted_id)

            entity = mock_core.MockBenchmark()
            benchmarks.append(zombie_db_client.mock.benchmarks_collection.insert_one(entity.serialize()).inserted_id)

        for system_id in systems:
            for image_source_id in image_sources:
                # Create trial results for each combination of systems and image sources,
                # as if the run system tasks are complete
                entity = arvet.core.trial_result.TrialResult(
                    system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                trial_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
                trial_results.append(trial_result_id)
                task = zombie_task_manager.get_run_system_task(system_id, image_source_id)
                task.mark_job_started('test', 0)
                task.mark_job_complete(trial_result_id)

        for trial_result_id in trial_results:
            for benchmark_id in benchmarks:
                # Create benchmark results for each combination of trial result and benchmark,
                # as if the benchmark system tasks are complete
                entity = arvet.core.benchmark.BenchmarkResult(benchmark_id, trial_result_id, True)
                benchmark_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
                benchmark_results.append(benchmark_result_id)
                task = zombie_task_manager.get_benchmark_task(trial_result_id, benchmark_id)
                task.mark_job_started('test', 0)
                task.mark_job_complete(benchmark_result_id)

        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, benchmarks)

        for trial_result_id in trial_results:
            for benchmark_id in benchmarks:
                benchmark_result_id = subject.get_benchmark_result(trial_result_id, benchmark_id)
                self.assertIsNotNone(benchmark_result_id)
                self.assertIn(benchmark_result_id, benchmark_results)

    def test_store_and_get_trial_result_basic(self):
        subject = MockExperiment()
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        trial_result_id = bson.ObjectId()
        subject.store_trial_result(system_id, image_source_id, trial_result_id)
        self.assertEqual([trial_result_id], subject.get_trial_results(system_id, image_source_id))

    def test_store_trial_result_persists_when_serialized(self):
        mock_db_client = self.create_mock_db_client()
        subject = MockExperiment()
        # Create and store a system, image source, and trial result in the database.
        # They should get removed if they don't exist.
        system_id = mock_db_client.system_collection.insert_one(mock_core.MockSystem().serialize()).inserted_id
        image_source_id = mock_db_client.image_source_collection.insert_one(
            mock_core.MockImageSource().serialize()).inserted_id
        trial_result_id = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
                    system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id
        subject.store_trial_result(system_id, image_source_id, trial_result_id)

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()
        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertEqual([trial_result_id], subject.get_trial_results(system_id, image_source_id))

    def test_deserialize_clears_invalid_trials_from_trial_map(self):
        mock_db_client = self.create_mock_db_client()
        missing_system = bson.ObjectId()
        missing_image_source = bson.ObjectId()
        missing_trial = bson.ObjectId()

        # Add some descendant objects, which should not on their own be removed from the map
        missing_system_trial = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            missing_system, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id
        missing_source_trial = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            self.systems[0].identifier, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id

        subject = self.make_instance(trial_map={
            missing_system: {self.image_sources[0].identifier: [missing_system_trial]},
            self.systems[0].identifier: {missing_image_source: [missing_source_trial],
                                         self.image_sources[0].identifier: [missing_trial],
                                         self.image_sources[1].identifier: [self.trial_results[0].identifier],
                                         self.image_sources[2].identifier: [self.trial_results[0].identifier,
                                                                            missing_trial]}
        })

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()
        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertEqual([], subject.get_trial_results(missing_system, self.image_sources[0].identifier))
        self.assertEqual([], subject.get_trial_results(self.systems[0].identifier, missing_image_source))
        self.assertEqual([], subject.get_trial_results(self.systems[0].identifier, self.image_sources[0].identifier))
        self.assertEqual([self.trial_results[0].identifier],
                         subject.get_trial_results(self.systems[0].identifier, self.image_sources[1].identifier))
        self.assertEqual([self.trial_results[0].identifier],
                         subject.get_trial_results(self.systems[0].identifier, self.image_sources[2].identifier))

    def test_store_and_get_benchmark_result_basic(self):
        subject = MockExperiment()
        trial_result_id = bson.ObjectId()
        benchmark_id = bson.ObjectId()
        benchmark_result_id = bson.ObjectId()
        subject.store_benchmark_result(trial_result_id, benchmark_id, benchmark_result_id)
        self.assertEqual(benchmark_result_id, subject.get_benchmark_result(trial_result_id, benchmark_id))

    def test_store_benchmark_result_persists_when_serialized(self):
        mock_db_client = self.create_mock_db_client()
        subject = MockExperiment()
        trial_result_id = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            self.systems[0].identifier, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id
        benchmark_id = mock_db_client.benchmarks_collection.insert_one(
            mock_core.MockBenchmark().serialize()).inserted_id
        benchmark_result_id = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(benchmark_id, trial_result_id, True).serialize()).inserted_id
        subject.store_benchmark_result(trial_result_id, benchmark_id, benchmark_result_id)

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()
        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertEqual(benchmark_result_id, subject.get_benchmark_result(trial_result_id, benchmark_id))

    def test_deserialize_clears_invalid_results_from_result_map(self):
        mock_db_client = self.create_mock_db_client()
        missing_trial = bson.ObjectId()
        missing_benchmark = bson.ObjectId()
        missing_benchmark_result = bson.ObjectId()

        # Add some descendant objects, which should not on their own be removed from the map
        result_missing_trial = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(self.benchmarks[0].identifier, missing_trial, True).serialize()
        ).inserted_id
        result_missing_benchmark = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(missing_benchmark, self.trial_results[0].identifier, True).serialize()
        ).inserted_id
        self.assertIsInstance(result_missing_trial, bson.ObjectId)
        self.assertIsInstance(result_missing_benchmark, bson.ObjectId)

        subject = self.make_instance(result_map={
            missing_trial: {self.benchmarks[0].identifier: result_missing_trial},
            self.trial_results[0].identifier: {missing_benchmark: result_missing_benchmark,
                                               self.benchmarks[0].identifier: missing_benchmark_result,
                                               self.benchmarks[1].identifier: self.benchmark_results[0].identifier}
        })

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()

        # Make sure the test state has been preserved through serialize
        self.assertIn(str(missing_trial), s_subject['result_map'])
        self.assertIn(str(self.benchmarks[0].identifier), s_subject['result_map'][str(missing_trial)])
        self.assertEqual(result_missing_trial,
                         s_subject['result_map'][str(missing_trial)][str(self.benchmarks[0].identifier)])

        self.assertIn(str(self.trial_results[0].identifier), s_subject['result_map'])
        self.assertIn(str(missing_benchmark), s_subject['result_map'][str(self.trial_results[0].identifier)])
        self.assertEqual(result_missing_benchmark,
                         s_subject['result_map'][str(self.trial_results[0].identifier)][str(missing_benchmark)])
        self.assertIn(str(self.benchmarks[0].identifier),
                      s_subject['result_map'][str(self.trial_results[0].identifier)])
        self.assertEqual(missing_benchmark_result,
                         s_subject['result_map'][str(self.trial_results[0].identifier)][str(self.benchmarks[0].identifier)])
        self.assertIn(str(self.benchmarks[1].identifier), s_subject['result_map'][str(self.trial_results[0].identifier)])
        self.assertEqual(self.benchmark_results[0].identifier,
                         s_subject['result_map'][str(self.trial_results[0].identifier)][str(self.benchmarks[1].identifier)])

        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertIsNone(subject.get_benchmark_result(missing_trial, self.image_sources[0].identifier))
        self.assertIsNone(subject.get_benchmark_result(self.trial_results[0].identifier, missing_benchmark))
        self.assertIsNone(subject.get_benchmark_result(self.trial_results[0].identifier, self.benchmarks[0].identifier))
        self.assertIsNotNone(subject.get_benchmark_result(self.trial_results[0].identifier,
                                                          self.benchmarks[1].identifier))
