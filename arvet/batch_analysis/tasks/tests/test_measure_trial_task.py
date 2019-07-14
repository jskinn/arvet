# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import logging
import bson
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
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

    @mock.patch('arvet.batch_analysis.tasks.measure_trial_task.autoload_modules')
    def test_load_referenced_models_loads_metric_and_trial_models(self, mock_autoload):
        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.UNSTARTED
        )
        obj.save()

        # Delete and reload the object to reset the references to object ids
        del obj
        obj = next(Task.objects.all())

        # Autoload the model types
        self.assertFalse(mock_autoload.called)
        obj.load_referenced_modules()
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(Metric, [self.metric.pk]), mock_autoload.call_args_list)
        self.assertIn(mock.call(TrialResult, [self.trial_result.pk]), mock_autoload.call_args_list)

    def test_result_id_gets_id_without_dereferencing(self):
        result = InitMonitoredResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True
        )
        result.save()
        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.DONE,
            result=result
        )
        obj.save()

        # Set up mocks
        dereferenced = False

        def init_side_effect(_):
            nonlocal dereferenced
            dereferenced = True
        InitMonitoredResult.side_effect = init_side_effect

        # Delete and reload the object to reset the references to object ids
        del obj
        obj = next(Task.objects.all())

        # Autoload the model types
        _ = obj.result_id
        self.assertFalse(dereferenced)

        # Clean up
        InitMonitoredResult.side_effect = None

    @mock.patch('arvet.batch_analysis.tasks.measure_trial_task.autoload_modules')
    def test_get_result_autoloads_model_type_before_dereferencing(self, mock_autoload):
        # Set up objects
        result = InitMonitoredResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True
        )
        result.save()

        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.DONE,
            result=result
        )
        obj.save()

        # Set up mocks
        loaded = False
        constructed = False
        loaded_first = False

        def autoload_side_effect(model, *_, **__):
            nonlocal loaded
            if model == MetricResult:
                loaded = True
        mock_autoload.side_effect = autoload_side_effect

        def init_result_side_effect(_):
            nonlocal loaded, constructed, loaded_first
            constructed = True
            if loaded:
                loaded_first = True
        InitMonitoredResult.side_effect = init_result_side_effect

        # Delete and reload the object to reset the references to object ids
        del obj
        obj = next(Task.objects.all())

        # get the result
        obj.get_result()
        self.assertTrue(mock_autoload.called)
        self.assertEqual(mock.call(MetricResult, [result.pk]), mock_autoload.call_args)
        self.assertTrue(constructed)
        self.assertTrue(loaded)
        self.assertTrue(loaded_first)

        # Clean up
        InitMonitoredResult.side_effect = None


class InitMonitoredResult(mock_types.MockMetricResult):
    side_effect = None

    def __init__(self, *args, **kwargs):
        super(InitMonitoredResult, self).__init__(*args, **kwargs)
        if self.side_effect is not None:
            self.side_effect()


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
        metric_result = mock.create_autospec(mock_types.MockMetricResult)
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

    def test_run_task_saves_result(self):
        metric_result = mock.create_autospec(mock_types.MockMetricResult)
        self.metric.measure_results = lambda _: metric_result
        subject = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(metric_result.save.called)
        subject.run_task(self.path_manager)
        self.assertTrue(metric_result.save.called)

    def test_result_id_is_none_if_result_is_none(self):
        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.UNSTARTED
        )
        self.assertIsNone(obj.result_id)

    def test_result_id_is_result_primary_key(self):
        result = mock_types.MockMetricResult()
        result.pk = bson.ObjectId()
        obj = MeasureTrialTask(
            metric=self.metric,
            trial_results=[self.trial_result],
            state=JobState.DONE,
            result=result
        )
        self.assertEqual(result.pk, obj.result_id)
