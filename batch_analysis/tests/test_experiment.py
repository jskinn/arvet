import unittest
import unittest.mock as mock
import pymongo
import bson.objectid as oid
import util.dict_utils as du
import database.tests.test_entity
import database.client
import batch_analysis.job_system
import batch_analysis.experiment as ex


class TestExperiment(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return ex.Experiment

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'image_sources': {oid.ObjectId(), oid.ObjectId(), oid.ObjectId()},
            'systems': {oid.ObjectId(), oid.ObjectId(), oid.ObjectId()},
            'benchmarks': {oid.ObjectId(), oid.ObjectId()},
            'trial_results': {oid.ObjectId()},
            'benchmark_results': {oid.ObjectId()}
        })
        if 'trial_map' not in kwargs:
            states = [0, 0, 2, 1, 0, 1, 1, 0, 0]
            kwargs['trial_map'] = {
                sys_id: {source_id: ex.ProgressState(states[(7 * idx) % len(states)])
                         for idx, source_id in enumerate(kwargs['image_sources'])}
                for sys_id in kwargs['systems']
            }
        if 'benchmark_map' not in kwargs:
            states = [2, 0, 1, 0, 0, 0, 2, 1, 1]
            kwargs['benchmark_map'] = {
                trial_id: {bench_id: ex.ProgressState(states[(13 * idx) % len(states)])
                           for idx, bench_id in enumerate(kwargs['benchmarks'])}
                for trial_id in kwargs['trial_results']
            }
        return ex.Experiment(*args, **kwargs)

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
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1._image_sources, experiment2._image_sources)
        self.assertEqual(experiment1._systems, experiment2._systems)
        self.assertEqual(experiment1._benchmarks, experiment2._benchmarks)
        self.assertEqual(experiment1._trial_results, experiment2._trial_results)
        self.assertEqual(experiment1._trial_map, experiment2._trial_map)
        self.assertEqual(experiment1._benchmark_map, experiment2._benchmark_map)

    def assert_serialized_equal(self, s_model1, s_model2):
        # Check that either both or neither have an id
        if '_id' in s_model1 and '_id' in s_model2:
            self.assertEqual(s_model1['_id'], s_model2['_id'])
        else:
            self.assertNotIn('_id', s_model1)
            self.assertNotIn('_id', s_model2)
        for key in {'trial_map', 'benchmark_map'}:
            self.assertEqual(s_model1[key], s_model2[key], "Values for {0} were not equal".format(key))
        # Special handling for set keys, since the order is allowed to change.
        for key in {'image_sources', 'systems', 'trial_results', 'benchmarks', 'benchmark_results'}:
            self.assertEqual(set(s_model1[key]), set(s_model2[key]), "Values for {0} were not equal".format(key))

    def test_constructor_works_with_minimal_arguments(self):
        ex.Experiment()

    def test_do_imports_imports_systems_sources_and_benchmarks(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        subject = ex.Experiment(id_=oid.ObjectId())
        subject.import_systems = mock.create_autospec(subject.import_systems)
        subject.import_systems.return_value = {system_id}
        subject.import_image_sources = mock.create_autospec(subject.import_image_sources)
        subject.import_image_sources.return_value = {image_source_id}
        subject.import_benchmarks = mock.create_autospec(subject.import_benchmarks)
        subject.import_benchmarks.return_value = {benchmark_id}

        subject.do_imports(mock_db_client, save_changes=False)
        self.assertTrue(subject.import_systems.called)
        self.assertIn(system_id, subject._systems)
        self.assertTrue(subject.import_image_sources.called)
        self.assertIn(image_source_id, subject._image_sources)
        self.assertTrue(subject.import_benchmarks.called)
        self.assertIn(benchmark_id, subject._benchmarks)
        self.assertEqual({
            '$addToSet': {
                'systems': [system_id],
                'image_sources': [image_source_id],
                'benchmarks': [benchmark_id]
            }
        }, subject._updates)

    def test_do_imports_handles_duplicates(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        system_id1 = oid.ObjectId()
        system_id2 = oid.ObjectId()
        image_source_id1 = oid.ObjectId()
        image_source_id2 = oid.ObjectId()
        benchmark_id1 = oid.ObjectId()
        benchmark_id2 = oid.ObjectId()
        subject = ex.Experiment(id_=oid.ObjectId(), systems={system_id1},
                                image_sources={image_source_id1}, benchmarks={benchmark_id1})
        subject.import_systems = mock.create_autospec(subject.import_systems)
        subject.import_systems.return_value = {system_id1, system_id2}
        subject.import_image_sources = mock.create_autospec(subject.import_image_sources)
        subject.import_image_sources.return_value = {image_source_id1, image_source_id2}
        subject.import_benchmarks = mock.create_autospec(subject.import_benchmarks)
        subject.import_benchmarks.return_value = {benchmark_id1, benchmark_id2}

        subject.do_imports(mock_db_client, save_changes=False)
        self.assertEqual({system_id1, system_id2}, subject._systems)
        self.assertEqual({image_source_id1, image_source_id2}, subject._image_sources)
        self.assertEqual({benchmark_id1, benchmark_id2}, subject._benchmarks)
        self.assertEqual({
            '$addToSet': {
                'systems': [system_id2],
                'image_sources': [image_source_id2],
                'benchmarks': [benchmark_id2]
            }
        }, subject._updates)

    def test_do_imports_can_save_changes(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        subject = ex.Experiment(id_=oid.ObjectId())
        subject.import_systems = mock.create_autospec(subject.import_systems)
        subject.import_systems.return_value = {system_id}
        subject.import_image_sources = mock.create_autospec(subject.import_image_sources)
        subject.import_image_sources.return_value = {image_source_id}
        subject.import_benchmarks = mock.create_autospec(subject.import_benchmarks)
        subject.import_benchmarks.return_value = {benchmark_id}

        subject.do_imports(mock_db_client, save_changes=True)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
            '_id': subject.identifier
        }, {
            '$addToSet': {
                'systems': [system_id],
                'image_sources': [image_source_id],
                'benchmarks': [benchmark_id]
            }
        }), mock_db_client.experiments_collection.update.call_args)

    def test_schedule_trials_updates_trial_map(self):
        mock_job_system = mock.create_autospec(batch_analysis.job_system.JobSystem)
        image_source_ids = (oid.ObjectId(), oid.ObjectId())
        system_ids = (oid.ObjectId(), oid.ObjectId())
        subject = ex.Experiment(id_=oid.ObjectId(), image_sources=image_source_ids, systems=system_ids)
        subject.schedule_tasks(mock_job_system)
        self.assertEqual({
            system_ids[0]: {
                image_source_ids[0]: ex.ProgressState.RUNNING,
                image_source_ids[1]: ex.ProgressState.RUNNING,
            },
            system_ids[1]: {
                image_source_ids[0]: ex.ProgressState.RUNNING,
                image_source_ids[1]: ex.ProgressState.RUNNING,
            }
        }, subject._trial_map)

    def test_schedule_trials_updates_benchmark_map(self):
        mock_job_system = mock.create_autospec(batch_analysis.job_system.JobSystem)
        benchmark_ids = (oid.ObjectId(), oid.ObjectId())
        trial_result_ids = (oid.ObjectId(), oid.ObjectId())
        subject = ex.Experiment(id_=oid.ObjectId(), benchmarks=benchmark_ids, trial_results=trial_result_ids)
        subject.schedule_tasks(mock_job_system)
        self.assertEqual({
            trial_result_ids[0]: {
                benchmark_ids[0]: ex.ProgressState.RUNNING,
                benchmark_ids[1]: ex.ProgressState.RUNNING,
            },
            trial_result_ids[1]: {
                benchmark_ids[0]: ex.ProgressState.RUNNING,
                benchmark_ids[1]: ex.ProgressState.RUNNING,
            }
        }, subject._benchmark_map)

    def test_schedule_trials_runs_systems(self):
        mock_job_system = mock.create_autospec(batch_analysis.job_system.JobSystem)
        image_source_ids = (oid.ObjectId(), oid.ObjectId())
        system_ids = (oid.ObjectId(), oid.ObjectId())
        subject = ex.Experiment(id_=oid.ObjectId(), image_sources=image_source_ids, systems=system_ids,
                                trial_map={system_ids[0]: {image_source_ids[0]: ex.ProgressState.RUNNING},
                                           system_ids[1]: {image_source_ids[0]: ex.ProgressState.UNSTARTED}})
        subject.schedule_tasks(mock_job_system)
        self.assertTrue(mock_job_system.schedule_run_system.called)
        self.assertNotIn(mock.call(system_ids[0], image_source_ids[0], subject.identifier),
                         mock_job_system.schedule_run_system.call_args_list)
        self.assertIn(mock.call(system_ids[0], image_source_ids[1], subject.identifier),
                      mock_job_system.schedule_run_system.call_args_list)
        self.assertIn(mock.call(system_ids[1], image_source_ids[0], subject.identifier),
                      mock_job_system.schedule_run_system.call_args_list)
        self.assertIn(mock.call(system_ids[0], image_source_ids[1], subject.identifier),
                      mock_job_system.schedule_run_system.call_args_list)

    def test_schedule_trials_performs_benchmarks(self):
        mock_job_system = mock.create_autospec(batch_analysis.job_system.JobSystem)
        benchmark_ids = (oid.ObjectId(), oid.ObjectId())
        trial_result_ids = (oid.ObjectId(), oid.ObjectId())
        subject = ex.Experiment(id_=oid.ObjectId(), benchmarks=benchmark_ids, trial_results=trial_result_ids,
                                benchmark_map={trial_result_ids[0]: {benchmark_ids[0]: ex.ProgressState.RUNNING},
                                               trial_result_ids[1]: {benchmark_ids[0]: ex.ProgressState.UNSTARTED}})
        subject.schedule_tasks(mock_job_system)
        self.assertTrue(mock_job_system.schedule_benchmark_result.called)
        self.assertNotIn(mock.call(trial_result_ids[0], benchmark_ids[0], subject.identifier),
                         mock_job_system.schedule_benchmark_result.call_args_list)
        self.assertIn(mock.call(trial_result_ids[0], benchmark_ids[1], subject.identifier),
                      mock_job_system.schedule_benchmark_result.call_args_list)
        self.assertIn(mock.call(trial_result_ids[1], benchmark_ids[0], subject.identifier),
                      mock_job_system.schedule_benchmark_result.call_args_list)
        self.assertIn(mock.call(trial_result_ids[0], benchmark_ids[1], subject.identifier),
                      mock_job_system.schedule_benchmark_result.call_args_list)

    def test_schedule_trials_can_save_updates(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_job_system = mock.create_autospec(batch_analysis.job_system.JobSystem)
        image_source_ids = (oid.ObjectId(), oid.ObjectId())
        system_ids = (oid.ObjectId(), oid.ObjectId())
        benchmark_ids = (oid.ObjectId(), oid.ObjectId())
        trial_result_ids = (oid.ObjectId(), oid.ObjectId())
        subject = ex.Experiment(id_=oid.ObjectId(), image_sources=image_source_ids, systems=system_ids,
                                benchmarks=benchmark_ids, trial_results=trial_result_ids,
                                trial_map={system_ids[0]: {image_source_ids[0]: ex.ProgressState.RUNNING},
                                           system_ids[1]: {image_source_ids[0]: ex.ProgressState.UNSTARTED}},
                                benchmark_map={trial_result_ids[0]: {benchmark_ids[0]: ex.ProgressState.RUNNING},
                                               trial_result_ids[1]: {benchmark_ids[0]: ex.ProgressState.UNSTARTED}})
        subject.schedule_tasks(mock_job_system, db_client=mock_db_client)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
            '_id': subject.identifier
        }, {
            '$set': {
                "trial_map.{0}.{1}".format(str(system_ids[0]), str(image_source_ids[1])): 1,
                "trial_map.{0}.{1}".format(str(system_ids[1]), str(image_source_ids[0])): 1,
                "trial_map.{0}.{1}".format(str(system_ids[1]), str(image_source_ids[1])): 1,
                "benchmark_map.{0}.{1}".format(str(trial_result_ids[0]), str(benchmark_ids[1])): 1,
                "benchmark_map.{0}.{1}".format(str(trial_result_ids[1]), str(benchmark_ids[0])): 1,
                "benchmark_map.{0}.{1}".format(str(trial_result_ids[1]), str(benchmark_ids[1])): 1,
            },
        }), mock_db_client.experiments_collection.update.call_args)

    def test_retry_trial_resets_progress(self):
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        subject = ex.Experiment(systems={system_id}, image_sources={image_source_id}, trial_map={
            system_id: {image_source_id: ex.ProgressState.RUNNING}
        })
        subject.retry_trial(system_id, image_source_id)
        self.assertEqual(ex.ProgressState.UNSTARTED, subject._trial_map[system_id][image_source_id])

    def test_retry_trial_saves_changes_if_given_db_client(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)

        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        subject = ex.Experiment(systems={system_id}, image_sources={image_source_id}, trial_map={
            system_id: {image_source_id: ex.ProgressState.RUNNING}
        }, id_=oid.ObjectId())

        subject.retry_trial(system_id, image_source_id, mock_db_client)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
             '_id': subject.identifier
         }, {
            '$set': {"trial_map.{0}.{1}".format(str(system_id), str(image_source_id)): 0}
        }), mock_db_client.experiments_collection.update.call_args)

    def test_add_trial_result_updates_progress(self):
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        subject = ex.Experiment(systems={system_id}, image_sources={image_source_id}, trial_map={
            system_id: {image_source_id: ex.ProgressState.RUNNING}
        })
        subject.add_trial_result(system_id, image_source_id, oid.ObjectId())
        self.assertEqual(ex.ProgressState.FINISHED, subject._trial_map[system_id][image_source_id])

    def test_add_trial_result_saves_changes_if_given_db_client(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)

        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        trial_result_id = oid.ObjectId()
        subject = ex.Experiment(systems={system_id}, image_sources={image_source_id}, benchmarks={benchmark_id},
                                trial_map={system_id: {image_source_id: ex.ProgressState.RUNNING}}, id_=oid.ObjectId())

        subject.add_trial_result(system_id, image_source_id, trial_result_id, mock_db_client)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
             '_id': subject.identifier
         }, {
            '$set': {
                "trial_map.{0}.{1}".format(str(system_id), str(image_source_id)): 2,
                "benchmark_map.{0}.{1}".format(str(trial_result_id), str(benchmark_id)): 0
            },
            '$addToSet': {'trial_results': [trial_result_id]}
        }), mock_db_client.experiments_collection.update.call_args)

    def test_retry_benchmark_resets_progress(self):
        trial_result_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        subject = ex.Experiment(trial_results={trial_result_id}, benchmarks={benchmark_id}, benchmark_map={
            trial_result_id: {benchmark_id: ex.ProgressState.RUNNING}
        })
        subject.retry_benchmark(trial_result_id, benchmark_id)
        self.assertEqual(ex.ProgressState.UNSTARTED, subject._benchmark_map[trial_result_id][benchmark_id])

    def test_retry_benchmark_saves_changes_if_given_db_client(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)

        trial_result_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        subject = ex.Experiment(trial_results={trial_result_id}, benchmarks={benchmark_id}, benchmark_map={
            trial_result_id: {benchmark_id: ex.ProgressState.RUNNING}
        }, id_=oid.ObjectId())

        subject.retry_benchmark(trial_result_id, benchmark_id, mock_db_client)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
             '_id': subject.identifier
         }, {
            '$set': {"benchmark_map.{0}.{1}".format(str(trial_result_id), str(benchmark_id)): 0}
        }), mock_db_client.experiments_collection.update.call_args)

    def test_add_benchmark_result_updates_progress(self):
        trial_result_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        subject = ex.Experiment(trial_results={trial_result_id}, benchmarks={benchmark_id}, benchmark_map={
            trial_result_id: {benchmark_id: ex.ProgressState.RUNNING}
        })
        subject.add_benchmark_result(trial_result_id, benchmark_id, oid.ObjectId())
        self.assertEqual(ex.ProgressState.FINISHED, subject._benchmark_map[trial_result_id][benchmark_id])

    def test_add_benchmark_result_saves_changes_if_given_db_client(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)

        benchmark_id = oid.ObjectId()
        trial_result_id = oid.ObjectId()
        benchmark_result_id = oid.ObjectId()
        subject = ex.Experiment(trial_results={trial_result_id}, benchmarks={benchmark_id}, id_=oid.ObjectId(),
                                benchmark_map={trial_result_id: {benchmark_id: ex.ProgressState.RUNNING}})

        subject.add_benchmark_result(trial_result_id, benchmark_id, benchmark_result_id, mock_db_client)
        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
             '_id': subject.identifier
         }, {
            '$set': {
                "benchmark_map.{0}.{1}".format(str(trial_result_id), str(benchmark_id)): 2
            },
            '$addToSet': {'benchmark_results': [benchmark_result_id]}
        }), mock_db_client.experiments_collection.update.call_args)

    def test_save_updates_inserts_if_no_id(self):
        new_id = oid.ObjectId()
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_db_client.experiments_collection.insert.return_value = new_id

        subject = ex.Experiment()
        subject._systems = {oid.ObjectId(), oid.ObjectId(), oid.ObjectId()}
        subject._image_sources = {oid.ObjectId()}
        subject._benchmarks = {oid.ObjectId(), oid.ObjectId()}
        subject._trial_results = {oid.ObjectId(), oid.ObjectId()}
        subject._benchmark_results = {oid.ObjectId(), oid.ObjectId(), oid.ObjectId(), oid.ObjectId()}
        subject.save_updates(mock_db_client)

        self.assertTrue(mock_db_client.experiments_collection.insert.called)
        s_subject = subject.serialize()
        del s_subject['_id']    # This key didn't exist when it was first serialized
        self.assert_serialized_equal(s_subject, mock_db_client.experiments_collection.insert.call_args[0][0])
        self.assertEqual(new_id, subject.identifier)

    def test_save_updates_stores_accumulated_changes(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        mock_job_system = mock.create_autospec(batch_analysis.job_system.JobSystem)

        image_source_id = oid.ObjectId()
        image_source_id2 = oid.ObjectId()
        system_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        benchmark_id2 = oid.ObjectId()
        trial_result_id = oid.ObjectId()
        benchmark_result_id = oid.ObjectId()

        subject = ex.Experiment(
            id_=oid.ObjectId(),
            image_sources={image_source_id, image_source_id2},
            systems={system_id},
            benchmarks={benchmark_id, benchmark_id2},
            trial_map={system_id: {image_source_id: ex.ProgressState.RUNNING}}
        )
        subject.add_trial_result(system_id, image_source_id, trial_result_id)
        subject.schedule_tasks(mock_job_system)
        subject.add_benchmark_result(trial_result_id, benchmark_id, benchmark_result_id)
        subject.save_updates(mock_db_client)

        self.assertTrue(mock_db_client.experiments_collection.update.called)
        self.assertEqual(mock.call({
            '_id': subject.identifier
        }, {
            '$set': {
                "trial_map.{0}.{1}".format(str(system_id), str(image_source_id)): 2,
                "trial_map.{0}.{1}".format(str(system_id), str(image_source_id2)): 1,
                "benchmark_map.{0}.{1}".format(str(trial_result_id), str(benchmark_id)): 2,
                "benchmark_map.{0}.{1}".format(str(trial_result_id), str(benchmark_id2)): 1
            },
            '$addToSet': {
                'trial_results': [trial_result_id],
                'benchmark_results': [benchmark_result_id]
            }
        }), mock_db_client.experiments_collection.update.call_args)

    def test_save_updates_does_nothing_if_no_changes(self):
        mock_db_client = mock.create_autospec(database.client.DatabaseClient)
        mock_db_client.experiments_collection = mock.create_autospec(pymongo.collection.Collection)
        subject = ex.Experiment(id_=oid.ObjectId())
        subject.save_updates(mock_db_client)
        self.assertFalse(mock_db_client.experiments_collection.update.called)
