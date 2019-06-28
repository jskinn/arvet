# Copyright (c) 2017, John Skinner
import unittest
import logging
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask


class TestMeasureTrialTaskDatabase(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    trial_result = None
    metric_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.metric = mock_types.MockMetric()
        cls.system.save()
        cls.image_source.save()
        cls.metric.save()

        cls.trial_result = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

        cls.metric_result = mock_types.MockMetricResult(metric=cls.metric, trial_results=[cls.trial_result],
                                                        success=True)
        cls.metric_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        mock_types.MockMetricResult._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_stores_and_loads_unstarted(self):
        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.UNSTARTED
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_completed(self):
        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
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
        obj = MeasureTrialTask(
            # metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = MeasureTrialTask(
            metric=self.metric,
            # trial_results=[self.trial_result],
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result]
            # state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()


class TestMeasureTrialTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        self.path_manager = PathManager(['~'])
        self.metric = mock_types.MockMetric()
        self.trial_result = mock_types.MockTrialResult(system=system, image_source=image_source, success=False)

    def test_run_task_records_unable_to_measure_trial(self):
        self.metric.is_trial_appropriate = lambda _: False
        subject = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
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
        self.assertEqual([self.trial_result], subject.result.trial_results)

    def test_run_task_records_exception_during_execution_and_re_raises(self):
        message = 'No mercy. No respite.'

        def bad_measure_results(*_, **__):
            raise ValueError(message)

        self.metric.measure_results = bad_measure_results
        subject = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
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
        self.assertEqual([self.trial_result], subject.result.trial_results)

    def test_run_task_records_metric_returned_none(self):
        self.metric.measure_results = lambda _: None
        subject = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
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
        self.assertEqual([self.trial_result], subject.result.trial_results)

    def test_run_task_records_returned_metric_result(self):
        metric_result = mock_types.MockMetricResult(metric=self.metric, trial_results=[self.trial_result], success=True)
        self.metric.measure_results = lambda _: metric_result
        subject = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertEqual(subject.result, metric_result)
