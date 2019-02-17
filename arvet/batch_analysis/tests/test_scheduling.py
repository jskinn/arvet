# Copyright (c) 2017, John Skinner
import unittest
import bson
import arvet.database.tests.database_connection as dbconn
from arvet.core.trial_result import TrialResult
from arvet.core.metric import MetricResult
from arvet.core.trial_comparison import TrialComparisonResult
import arvet.core.tests.mock_types as mock_types

from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask

import arvet.batch_analysis.task_manager as task_manager

import arvet.core.trial_result
import arvet.core.benchmark
import arvet.core.sequence_type

import arvet.batch_analysis.scheduling as scheduling


class TestScheduleAll(unittest.TestCase):
    systems = None
    image_sources = None
    metrics = None
    comparison_metrics = None
    trial_result_1 = None
    trial_result_2 = None
    metric_result = None
    comaprison_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.systems = [mock_types.MockSystem() for _ in range(2)]

        cls.image_sources = [mock_types.MockImageSource() for _ in range(2)]
        cls.metrics = [mock_types.MockMetric() for _ in range(2)]
        cls.comparison_metrics = [mock_types.MockTrialComparisonMetric() for _ in range(2)]

        for system in cls.systems:
            system.save()
        for image_source in cls.image_sources:
            image_source.save()
        for metric in cls.metrics:
            metric.save()
        for metric in cls.comparison_metrics:
            metric.save()

        # cls.trial_result_1 = TrialResult(image_source=cls.image_sources[0], system=cls.systems[0], success=True)
        # cls.trial_result_2 = TrialResult(image_source=cls.image_sources[1], system=cls.systems[1], success=True)
        # cls.trial_result_1.save()
        # cls.trial_result_2.save()

        # cls.metric_result = MetricResult(metric=cls.metric[0], trial_results=[cls.trial_result_1], success=True)
        # cls.metric_result.save()
        # cls.comparison_result = TrialComparisonResult(
        #     metric=cls.comparison_metric,
        #     trial_results_1=[cls.trial_result_1],
        #     trial_results_2=[cls.trial_result_2],
        #     success=True)
        # cls.comparison_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        TrialResult._mongometa.collection.drop()
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        TrialComparisonResult._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_schedule_all_schedules_all_trial_combinations(self):
        scheduling.schedule_all(
            self.systems,
            self.image_sources,
            [],
            repeats=2
        )
        self.assertEqual(len(self.systems) * len(self.image_sources) * 2, RunSystemTask.objects.all().count())
        for system in self.systems:
            for image_source in self.image_sources:
                for repeat in range(2):
                    self.assertEqual(1, RunSystemTask.objects.raw({
                        'system': system.identifier,
                        'image_source': image_source.identifier,
                        'repeat': repeat
                    }).count())

    def test_schedule_all_doesnt_reschedule_complete_trials(self):
        for system in self.systems:
            for image_source in self.image_sources:
                trial_result = TrialResult(
                    image_source=image_source,
                    system=system,
                    success=True
                )
                trial_result.save()

                task = task_manager.get_run_system_task(
                    system=system,
                    image_source=image_source,
                    repeat=1
                )
                task.mark_job_started('test', 0)
                task.mark_job_complete()
                task.result = trial_result
                task.save()

        scheduling.schedule_all(
            self.systems,
            self.image_sources,
            [],
            repeats=2
        )
        self.assertEqual(len(self.systems) * len(self.image_sources) * 2, RunSystemTask.objects.all().count())
        for system in self.systems:
            for image_source in self.image_sources:
                self.assertEqual(1, RunSystemTask.objects.raw({
                    'system': system.identifier,
                    'image_source': image_source.identifier,
                    'repeat': 0,
                    'state': JobState.UNSTARTED.name
                }).count())
                self.assertEqual(1, RunSystemTask.objects.raw({
                    'system': system.identifier,
                    'image_source': image_source.identifier,
                    'repeat': 0,
                    'state': JobState.UNSTARTED.name
                }).count())

    def test_schedule_all_schedules_all_metric_combinations(self):
        # Add tasks indicating running the system is already done
        trial_result_groups = []
        for system in self.systems:
            for image_source in self.image_sources:
                trial_result_group = []
                for repeat in range(2):
                    trial_result = TrialResult(
                        image_source=image_source,
                        system=system,
                        success=True
                    )
                    trial_result.save()
                    trial_result_group.append(trial_result.identifier)

                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=repeat
                    )
                    task.mark_job_started('test', 0)
                    task.mark_job_complete()
                    task.result = trial_result
                    task.save()
                trial_result_groups.append(trial_result_group)

        scheduling.schedule_all(
            self.systems,
            self.image_sources,
            self.metrics,
            repeats=2
        )
        self.assertEqual(len(self.metrics) * len(trial_result_groups), MeasureTrialTask.objects.all().count())
        for metric_idx, metric in enumerate(self.metrics):
            for group_idx, trial_result_group in enumerate(trial_result_groups):
                self.assertEqual(1, MeasureTrialTask.objects.raw({
                    'metric': metric.identifier,
                    'trial_results': {'$all': trial_result_group}
                }).count(), "Missing task for metric {0} group {1}".format(metric_idx, group_idx))

    def test_schedule_all_doesnt_schedule_metrics_if_repeats_are_incomplete(self):
        complete_trial_result_groups = []
        incomplete_trial_result_groups = []

        # Create trial results for each combination of systems and image sources,
        # as if the run system tasks are complete
        num_runs = 1
        for system in self.systems:
            for image_source in self.image_sources:
                trial_result_group = []
                for repeat in range(num_runs):
                    trial_result = TrialResult(
                        image_source=image_source,
                        system=system,
                        success=True
                    )
                    trial_result.save()
                    trial_result_group.append(trial_result.identifier)

                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=repeat
                    )
                    task.mark_job_started('test', 0)
                    task.mark_job_complete()
                    task.result = trial_result
                    task.save()
                if num_runs < 3:
                    incomplete_trial_result_groups.append(trial_result_group)
                    num_runs += 1
                else:
                    complete_trial_result_groups.append(trial_result_group)
                    num_runs = 1

        scheduling.schedule_all(
            self.systems,
            self.image_sources,
            self.metrics,
            repeats=3
        )

        self.assertEqual(len(complete_trial_result_groups) * len(self.metrics), MeasureTrialTask.objects.all().count())

        for metric_idx, metric in enumerate(self.metrics):
            # Check if we scheduled all the benchmarks for the complete groups
            for group_idx, trial_result_group in enumerate(complete_trial_result_groups):
                self.assertEqual(1, MeasureTrialTask.objects.raw({
                    'metric': metric.identifier,
                    'trial_results': {'$all': trial_result_group}
                }).count(), "Missing task for metric {0} and complete group {1}".format(metric_idx, group_idx))

            # Check we did not schedule benchmarks for incomplete groups
            for group_idx, trial_result_group in enumerate(incomplete_trial_result_groups):
                self.assertEqual(0, MeasureTrialTask.objects.raw({
                    'metric': metric.identifier,
                    'trial_results': {'$all': trial_result_group}
                }).count(), "Got undesired task for metric {0} on incomplete group {1}".format(metric_idx, group_idx))

    def test_schedule_all_allow_metrics_on_incomplete_overrides_incomplete_groups(self):
        # Add tasks indicating running the system is already done
        trial_result_groups = []
        for system in self.systems:
            for image_source in self.image_sources:
                trial_result_group = []
                for repeat in range(2):
                    trial_result = TrialResult(
                        image_source=image_source,
                        system=system,
                        success=True
                    )
                    trial_result.save()
                    trial_result_group.append(trial_result.identifier)

                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=repeat
                    )
                    task.mark_job_started('test', 0)
                    task.mark_job_complete()
                    task.result = trial_result
                    task.save()
                trial_result_groups.append(trial_result_group)

        scheduling.schedule_all(
            systems=self.systems,
            image_sources=self.image_sources,
            metrics=self.metrics,
            repeats=3,
            allow_metrics_on_incomplete=True
        )

        # Check if we scheduled measurement of all the groups
        self.assertEqual(len(trial_result_groups) * len(self.metrics), MeasureTrialTask.objects.all().count())
        for metric_idx, metric in enumerate(self.metrics):
            for group_idx, trial_result_group in enumerate(trial_result_groups):
                self.assertEqual(1, MeasureTrialTask.objects.raw({
                    'metric': metric.identifier,
                    'trial_results': {'$all': trial_result_group}
                }).count(), "Missing task for metric {0} and group {1}".format(metric_idx, group_idx))

    def test_schedule_all_schedules_all_invalidates_and_redoes_metrics_with_changed_trials(self):
        # Create trial results for each combination of systems and image sources,
        # as if the run system tasks are complete
        self.skipTest('Not sure about this feature')
        trial_result_groups = []
        incomplete_metric_results = []
        incomplete_metric_tasks = []
        for system in self.systems:
            for image_source in self.image_sources:
                trial_result_group = []
                for repeat in range(2):
                    trial_result = TrialResult(
                        image_source=image_source,
                        system=system,
                        success=True
                    )
                    trial_result.save()
                    trial_result_group.append(trial_result.identifier)

                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=0
                    )
                    task.mark_job_started('test', 0)
                    task.mark_job_complete()
                    task.result = trial_result
                    task.save()
                trial_result_groups.append(trial_result_group)

                # Add incomplete tasks and results
                for metric in self.metrics:
                    metric_result = MetricResult(
                        metric=metric,
                        trial_results=[trial_result_group[0]],
                        success=True
                    )
                    metric_result.save()
                    incomplete_metric_results.append(metric_result.identifier)

                    task = task_manager.get_measure_trial_task(
                        trial_results=trial_result_group,
                        metric=metric
                    )
                    task.mark_job_started('test', 0)
                    task.mark_job_complete()
                    task.result = metric_result
                    task.save()
                    incomplete_metric_tasks.append(task.identifier)

        scheduling.schedule_all(
            systems=self.systems,
            image_sources=self.image_sources,
            metrics=self.metrics,
            repeats=2
        )

        # Check all the metric results and tasks are gone
        self.assertEqual(0, MetricResult.objects.raw({'_id': {'$in': incomplete_metric_results}}).count())
        self.assertEqual(0, MeasureTrialTask.objects.raw({'_id': {'$in': incomplete_metric_tasks}}).count())

        # Check if we scheduled measuring all the groups again
        self.assertEqual(len(trial_result_groups) * len(self.metrics), MeasureTrialTask.objects.all().count())
        for metric_idx, metric in enumerate(self.metrics):
            for group_idx, trial_result_group in enumerate(trial_result_groups):
                self.assertEqual(1, MeasureTrialTask.objects.raw({
                    'metric': metric.identifier,
                    'trial_results': {'$all': trial_result_group}
                }).count(), "Missing task for metric {0} and group {1}".format(metric_idx, group_idx))


class TestGetTrialResults(unittest.TestCase):

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
        system_id = mock_db_client.system_collection.insert_one(mock_types.MockSystem().serialize()).inserted_id
        image_source_id = mock_db_client.image_source_collection.insert_one(
            mock_types.MockImageSource().serialize()).inserted_id
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

        system_id = mock_db_client.system_collection.insert_one(mock_types.MockSystem().serialize()).inserted_id
        image_source_id = mock_db_client.image_source_collection.insert_one(
            mock_types.MockImageSource().serialize()).inserted_id
        trial_result_id = mock_db_client.trials_collection.insert_one(arvet.core.trial_result.TrialResult(
            system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
        ).serialize()).inserted_id
        benchmark_id = mock_db_client.benchmarks_collection.insert_one(
            mock_types.MockBenchmark().serialize()).inserted_id
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
