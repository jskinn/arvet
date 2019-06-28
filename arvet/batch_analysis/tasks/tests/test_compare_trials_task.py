# Copyright (c) 2017, John Skinner
import unittest
import logging
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet.core.trial_comparison import TrialComparisonResult
from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.compare_trials_task import CompareTrialTask


class TestCompareTrialsTaskDatabase(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    trial_result_1 = None
    trial_result_2 = None
    metric_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.metric = mock_types.MockTrialComparisonMetric()
        cls.system.save()
        cls.image_source.save()
        cls.metric.save()

        cls.trial_result_1 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_1.save()
        cls.trial_result_2.save()

        cls.metric_result = TrialComparisonResult(
            metric=cls.metric,
            trial_results_1=[cls.trial_result_1],
            trial_results_2=[cls.trial_result_2],
            success=True)
        cls.metric_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        TrialComparisonResult._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_stores_and_loads_unstarted(self):
        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.UNSTARTED
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_completed(self):
        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.DONE,
            result=self.metric_result
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_saving_throws_exeption_if_required_fields_are_missing(self):
        obj = CompareTrialTask(
            # metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = CompareTrialTask(
            metric=self.metric,
            # trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            # trial_results_2=[self.trial_result_2],
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2]
            # state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()


class TestCompareTrialsTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        self.path_manager = PathManager(['~'])
        self.metric = mock_types.MockTrialComparisonMetric()
        self.trial_result_1 = mock_types.MockTrialResult(system=system, image_source=image_source, success=False)
        self.trial_result_2 = mock_types.MockTrialResult(system=system, image_source=image_source, success=False)

    def test_run_task_records_unable_to_measure_trial_in_group_1(self):
        self.metric.is_trial_appropriate_for_first = lambda _: False
        subject = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertIsNotNone(subject.result)
        self.assertFalse(subject.result.success)
        self.assertIsNotNone(subject.result.message)
        self.assertEqual(self.metric, subject.result.metric)
        self.assertEqual([self.trial_result_1], subject.result.trial_results_1)
        self.assertEqual([self.trial_result_2], subject.result.trial_results_2)

    def test_run_task_records_unable_to_measure_trial_in_group_2(self):
        self.metric.is_trial_appropriate_for_second = lambda _: False
        subject = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertIsNotNone(subject.result)
        self.assertFalse(subject.result.success)
        self.assertIsNotNone(subject.result.message)
        self.assertEqual(self.metric, subject.result.metric)
        self.assertEqual([self.trial_result_1], subject.result.trial_results_1)
        self.assertEqual([self.trial_result_2], subject.result.trial_results_2)

    def test_run_task_records_exception_during_execution_and_re_raises(self):
        message = 'No mercy. No respite.'

        def bad_compare_trials(*_, **__):
            raise ValueError(message)

        self.metric.compare_trials = bad_compare_trials
        subject = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        with self.assertRaises(ValueError):
            subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertIsNotNone(subject.result)
        self.assertFalse(subject.result.success)
        self.assertIsNotNone(subject.result.message)
        self.assertIn(message, subject.result.message)
        self.assertEqual(self.metric, subject.result.metric)
        self.assertEqual([self.trial_result_1], subject.result.trial_results_1)
        self.assertEqual([self.trial_result_2], subject.result.trial_results_2)

    def test_run_task_records_metric_returned_none(self):
        self.metric.compare_trials = lambda *args, **kwargs: None
        subject = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertIsNotNone(subject.result)
        self.assertFalse(subject.result.success)
        self.assertIsNotNone(subject.result.message)
        self.assertEqual(self.metric, subject.result.metric)
        self.assertEqual([self.trial_result_1], subject.result.trial_results_1)
        self.assertEqual([self.trial_result_2], subject.result.trial_results_2)

    def test_run_task_records_returned_metric_result(self):
        comparison_result = TrialComparisonResult(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            success=True
        )
        self.metric.compare_trials = lambda *args, **kwargs: comparison_result
        subject = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertEqual(subject.result, comparison_result)
