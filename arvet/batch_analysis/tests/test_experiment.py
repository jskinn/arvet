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
        self.systems = [mock_core.MockSystem(id_=bson.ObjectId()) for _ in range(2)]
        self.image_sources = [mock_core.MockImageSource(id_=bson.ObjectId()) for _ in range(3)]
        self.benchmarks = [mock_core.MockBenchmark(id_=bson.ObjectId()) for _ in range(2)]
        self.trial_results = []
        self.benchmark_results = []
        self.trial_map = {}
        for system in self.systems:
            self.trial_map[system.identifier] = {}
            for image_source in self.image_sources:
                trial_result_group = [arvet.core.trial_result.TrialResult(
                    system_id=system.identifier,
                    success=True,
                    sequence_type=arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL,
                    system_settings={},
                    id_=bson.ObjectId())
                    for _ in range(3)]
                self.trial_results += trial_result_group
                results_group = {benchmark.identifier: arvet.core.benchmark.BenchmarkResult(
                    benchmark_id=benchmark.identifier,
                    trial_result_ids=trial_result_group,
                    success=True,
                    id_=bson.ObjectId())
                    for benchmark in self.benchmarks}
                self.benchmark_results += list(results_group.values())
                self.trial_map[system.identifier][image_source.identifier] = {
                    'trials': {obj.identifier for obj in trial_result_group},
                    'results': {bench_id: result.identifier for bench_id, result in results_group.items()}
                }

    def get_class(self):
        return MockExperiment

    def make_instance(self, *args, **kwargs):
        du.defaults(kwargs, {
            'enabled': False,
            'trial_map': self.trial_map
        })
        return MockExperiment(*args, **kwargs)

    def assert_models_equal(self, experiment1, experiment2):
        """
        Helper to assert that two experiments are equal
        We're going to violate encapsulation for a bit
        :param experiment1:
        :param experiment2:
        :return:
        """
        if (not isinstance(experiment1, ex.Experiment) or
                not isinstance(experiment2, ex.Experiment)):
            self.fail('object was not an Experiment')
        self.assertEqual(experiment1.enabled, experiment2.enabled)
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1._trial_map, experiment2._trial_map)

    def assert_serialized_equal(self, s_experiment1, s_experiment2):
        """

        :param s_experiment1:
        :param s_experiment2:
        :return:
        """
        self.assertEqual(set(s_experiment1.keys()), set(s_experiment2.keys()))

        for key in s_experiment1.keys():
            if key not in {'trial_map'}:
                self.assertEqual(s_experiment1[key], s_experiment2[key])

        # Special handling for the trial map because we don't care about the order of the trials list
        self.assertEqual(set(s_experiment1['trial_map'].keys()), set(s_experiment2['trial_map'].keys()))
        for sys_id in s_experiment1['trial_map'].keys():
            self.assertEqual(set(s_experiment1['trial_map'][sys_id].keys()),
                             set(s_experiment2['trial_map'][sys_id].keys()))
            for img_src_id in s_experiment1['trial_map'][sys_id]:
                self.assertEqual({'trials', 'results'}, set(s_experiment1['trial_map'][sys_id][img_src_id].keys()))
                self.assertEqual({'trials', 'results'}, set(s_experiment2['trial_map'][sys_id][img_src_id].keys()))
                self.assertEqual(s_experiment1['trial_map'][sys_id][img_src_id]['results'],
                                 s_experiment2['trial_map'][sys_id][img_src_id]['results'])
                self.assertEqual(set(s_experiment1['trial_map'][sys_id][img_src_id]['trials']),
                                 set(s_experiment2['trial_map'][sys_id][img_src_id]['trials']))

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
        trial_result_groups = []
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
                trial_result_group = set()
                for repeat in range(3):
                    entity = arvet.core.trial_result.TrialResult(
                        system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                    trial_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
                    trial_result_group.add(trial_result_id)
                    task = zombie_task_manager.get_run_system_task(system_id, image_source_id, repeat)
                    task.mark_job_started('test', 0)
                    task.mark_job_complete(trial_result_id)
                trial_result_groups.append(trial_result_group)

        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, benchmarks,
                             repeats=3)

        # Check if we scheduled all the benchmarks
        for trial_result_group in trial_result_groups:
            for benchmark_id in benchmarks:
                self.assertIn(mock.call(trial_result_ids=trial_result_group, benchmark_id=benchmark_id,
                                        memory_requirements=mock.ANY, expected_duration=mock.ANY),
                              zombie_task_manager.mock.get_benchmark_task.call_args_list)

    def test_schedule_all_doesnt_schedule_benchmarks_if_repeats_are_incomplete(self):
        zombie_db_client = mock_client_factory.create()
        zombie_task_manager = mock_manager_factory.create()
        subject = MockExperiment()
        systems = []
        image_sources = []
        complete_trial_result_groups = []
        incomplete_trial_result_groups = []
        benchmarks = []
        for _ in range(3):  # Create
            entity = mock_core.MockSystem()
            systems.append(zombie_db_client.mock.system_collection.insert_one(entity.serialize()).inserted_id)

            entity = mock_core.MockImageSource()
            image_sources.append(
                zombie_db_client.mock.image_source_collection.insert_one(entity.serialize()).inserted_id)

            entity = mock_core.MockBenchmark()
            benchmarks.append(zombie_db_client.mock.benchmarks_collection.insert_one(entity.serialize()).inserted_id)

        num_runs = 1
        for system_id in systems:
            for image_source_id in image_sources:
                # Create trial results for each combination of systems and image sources,
                # as if the run system tasks are complete
                trial_result_group = set()
                for repeat in range(num_runs):
                    entity = arvet.core.trial_result.TrialResult(
                        system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                    trial_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
                    trial_result_group.add(trial_result_id)
                    task = zombie_task_manager.get_run_system_task(system_id, image_source_id, repeat)
                    task.mark_job_started('test', 0)
                    task.mark_job_complete(trial_result_id)
                if num_runs < 3:
                    incomplete_trial_result_groups.append(trial_result_group)
                    num_runs += 1
                else:
                    complete_trial_result_groups.append(trial_result_group)
                    num_runs = 1

        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, benchmarks,
                             repeats=3)

        self.assertEqual(len(complete_trial_result_groups) * len(benchmarks),
                         zombie_task_manager.mock.get_benchmark_task.call_count)

        # Check if we scheduled all the benchmarks for the complete groups
        for trial_result_group in complete_trial_result_groups:
            for benchmark_id in benchmarks:
                self.assertIn(mock.call(trial_result_ids=trial_result_group, benchmark_id=benchmark_id,
                                        memory_requirements=mock.ANY, expected_duration=mock.ANY),
                              zombie_task_manager.mock.get_benchmark_task.call_args_list)

        # Check we did not schedule benchmarks for incomplete groups
        for trial_result_group in incomplete_trial_result_groups:
            for benchmark_id in benchmarks:
                self.assertNotIn(mock.call(trial_result_ids=trial_result_group, benchmark_id=benchmark_id,
                                           memory_requirements=mock.ANY, expected_duration=mock.ANY),
                                 zombie_task_manager.mock.get_benchmark_task.call_args_list)

    def test_schedule_all_allow_incomplete_benchmarks_overrides_incomplete_groups(self):
        zombie_db_client = mock_client_factory.create()
        zombie_task_manager = mock_manager_factory.create()
        subject = MockExperiment()
        systems = []
        image_sources = []
        trial_result_groups = []
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
                trial_result_group = set()
                for repeat in range(2):
                    entity = arvet.core.trial_result.TrialResult(
                        system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
                    trial_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
                    trial_result_group.add(trial_result_id)
                    task = zombie_task_manager.get_run_system_task(system_id, image_source_id, repeat)
                    task.mark_job_started('test', 0)
                    task.mark_job_complete(trial_result_id)
                trial_result_groups.append(trial_result_group)

        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, benchmarks,
                             repeats=3, allow_incomplete_benchmarks=True)

        # Check if we scheduled all the benchmarks
        for trial_result_group in trial_result_groups:
            for benchmark_id in benchmarks:
                self.assertIn(mock.call(trial_result_ids=trial_result_group, benchmark_id=benchmark_id,
                                        memory_requirements=mock.ANY, expected_duration=mock.ANY),
                              zombie_task_manager.mock.get_benchmark_task.call_args_list)

    def test_schedule_all_schedules_all_invalidates_and_redoes_benchmark_combinations_with_changed_trials(self):
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
                self.assertIn(mock.call(trial_result_ids={trial_result_id}, benchmark_id=benchmark_id,
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
                entity = arvet.core.benchmark.BenchmarkResult(benchmark_id, [trial_result_id], True)
                benchmark_result_id = zombie_db_client.mock.trials_collection.insert_one(entity.serialize()).inserted_id
                benchmark_results.append(benchmark_result_id)
                task = zombie_task_manager.get_benchmark_task([trial_result_id], benchmark_id)
                task.mark_job_started('test', 0)
                task.mark_job_complete(benchmark_result_id)

        subject.schedule_all(zombie_task_manager.mock, zombie_db_client.mock, systems, image_sources, benchmarks)

        for system_id in systems:
            for image_source_id in image_sources:
                for benchmark_id in benchmarks:
                    benchmark_result_id = subject.get_benchmark_result(system_id, image_source_id, benchmark_id)
                    self.assertIsNotNone(benchmark_result_id)
                    self.assertIn(benchmark_result_id, benchmark_results)

    def test_store_and_get_trial_result_basic(self):
        subject = MockExperiment()
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        trial_result_id = bson.ObjectId()
        mock_db_client = self.create_mock_db_client()
        subject.store_trial_results(system_id, image_source_id, [trial_result_id], mock_db_client)
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
        subject.store_trial_results(system_id, image_source_id, [trial_result_id], mock_db_client)

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()
        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertEqual({trial_result_id}, subject.get_trial_results(system_id, image_source_id))

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

        subject = self.make_instance()

        subject.store_trial_results(missing_system, self.image_sources[0].identifier, [missing_system_trial],
                                    mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, missing_image_source, [missing_source_trial],
                                    mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, self.image_sources[0].identifier, [missing_trial],
                                    mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, self.image_sources[1].identifier,
                                    [self.trial_results[0].identifier], mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, self.image_sources[2].identifier,
                                    [self.trial_results[0].identifier, missing_trial], mock_db_client)

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()
        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertEqual(set(), subject.get_trial_results(missing_system, self.image_sources[0].identifier))
        self.assertEqual(set(), subject.get_trial_results(self.systems[0].identifier, missing_image_source))
        self.assertEqual(set(), subject.get_trial_results(self.systems[0].identifier, self.image_sources[0].identifier))
        self.assertEqual({self.trial_results[0].identifier},
                         subject.get_trial_results(self.systems[0].identifier, self.image_sources[1].identifier))
        self.assertEqual({self.trial_results[0].identifier},
                         subject.get_trial_results(self.systems[0].identifier, self.image_sources[2].identifier))

    def test_store_and_get_benchmark_result_basic(self):
        mock_db_client = self.create_mock_db_client()
        subject = MockExperiment()
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        trial_result_id = bson.ObjectId()
        benchmark_id = bson.ObjectId()
        benchmark_result_id = bson.ObjectId()
        subject.store_trial_results(system_id, image_source_id, [trial_result_id], mock_db_client)
        subject.store_benchmark_result(system_id, image_source_id, benchmark_id, benchmark_result_id)
        self.assertEqual(benchmark_result_id, subject.get_benchmark_result(system_id, image_source_id, benchmark_id))

    def test_cannot_store_benchmark_result_without_trials(self):
        subject = MockExperiment()
        system_id = bson.ObjectId()
        image_source_id = bson.ObjectId()
        benchmark_id = bson.ObjectId()
        benchmark_result_id = bson.ObjectId()
        subject.store_benchmark_result(system_id, image_source_id, benchmark_id, benchmark_result_id)
        self.assertIsNone(subject.get_benchmark_result(system_id, image_source_id, benchmark_id))

    def test_store_benchmark_result_persists_when_serialized(self):
        mock_db_client = self.create_mock_db_client()
        subject = MockExperiment()

        system_id = mock_db_client.system_collection.insert_one(mock_core.MockSystem().serialize()).inserted_id
        image_source_id = mock_db_client.image_source_collection.insert_one(
            mock_core.MockImageSource().serialize()).inserted_id
        trial_result_id = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id
        benchmark_id = mock_db_client.benchmarks_collection.insert_one(
            mock_core.MockBenchmark().serialize()).inserted_id
        benchmark_result_id = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(benchmark_id, [trial_result_id], True).serialize()).inserted_id

        subject.store_trial_results(system_id, image_source_id, [trial_result_id], mock_db_client)
        subject.store_benchmark_result(system_id, image_source_id, benchmark_id, benchmark_result_id)

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()
        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertEqual(benchmark_result_id, subject.get_benchmark_result(system_id, image_source_id, benchmark_id))

    def test_deserialize_clears_invalid_results_from_trial_map(self):
        mock_db_client = self.create_mock_db_client()
        missing_system = bson.ObjectId()
        missing_image_source = bson.ObjectId()
        missing_trial = bson.ObjectId()
        missing_benchmark = bson.ObjectId()
        missing_benchmark_result = bson.ObjectId()

        # Add some descendant objects, which should not on their own be removed from the map
        missing_system_trial = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            missing_system, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id
        missing_source_trial = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            self.systems[0].identifier, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id

        result_missing_system = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(self.benchmarks[0].identifier, [missing_system_trial],
                                                 True).serialize()
        ).inserted_id
        result_missing_source = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(self.benchmarks[0].identifier, [missing_source_trial],
                                                 True).serialize()
        ).inserted_id
        result_missing_trial = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(self.benchmarks[0].identifier, [missing_trial], True).serialize()
        ).inserted_id
        result_missing_benchmark = mock_db_client.results_collection.insert_one(
            arvet.core.benchmark.BenchmarkResult(missing_benchmark, [self.trial_results[0].identifier],
                                                 True).serialize()
        ).inserted_id
        self.assertIsInstance(result_missing_trial, bson.ObjectId)
        self.assertIsInstance(result_missing_benchmark, bson.ObjectId)

        subject = self.make_instance()

        subject.store_trial_results(missing_system, self.image_sources[0].identifier, [missing_system_trial],
                                    mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, missing_image_source, [missing_source_trial],
                                    mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, self.image_sources[0].identifier, [missing_trial],
                                    mock_db_client)
        subject.store_trial_results(self.systems[0].identifier, self.image_sources[1].identifier,
                                    [self.trial_results[0].identifier], mock_db_client)

        subject.store_benchmark_result(missing_system, self.image_sources[0].identifier, self.benchmarks[0].identifier,
                                       result_missing_system)
        subject.store_benchmark_result(self.systems[0].identifier, missing_image_source, self.benchmarks[0].identifier,
                                       result_missing_source)
        subject.store_benchmark_result(self.systems[0].identifier, self.image_sources[0].identifier,
                                       self.benchmarks[0].identifier, result_missing_trial)
        subject.store_benchmark_result(self.systems[0].identifier, self.image_sources[1].identifier,
                                       missing_benchmark, result_missing_benchmark)
        subject.store_benchmark_result(self.systems[0].identifier, self.image_sources[1].identifier,
                                       self.benchmarks[0].identifier, missing_benchmark_result)
        subject.store_benchmark_result(self.systems[0].identifier, self.image_sources[1].identifier,
                                       self.benchmarks[1].identifier, self.benchmark_results[0].identifier)

        # Serialize and then deserialize the experiment
        s_subject = subject.serialize()

        subject = MockExperiment.deserialize(s_subject, mock_db_client)

        self.assertIsNone(subject.get_benchmark_result(missing_system, self.image_sources[0].identifier,
                                                       self.benchmarks[0].identifier))
        self.assertIsNone(subject.get_benchmark_result(self.systems[0].identifier, missing_image_source,
                                                       self.benchmarks[0].identifier))
        self.assertIsNone(subject.get_benchmark_result(self.systems[0].identifier, self.image_sources[0].identifier,
                                                       self.benchmarks[0].identifier))
        self.assertIsNone(subject.get_benchmark_result(self.systems[0].identifier, self.image_sources[1].identifier,
                                                       missing_benchmark))
        self.assertIsNone(subject.get_benchmark_result(self.systems[0].identifier, self.image_sources[1].identifier,
                                                       self.benchmarks[0].identifier))
        self.assertEqual(self.benchmark_results[0].identifier,
                         subject.get_benchmark_result(self.systems[0].identifier, self.image_sources[1].identifier,
                                                      self.benchmarks[1].identifier))
