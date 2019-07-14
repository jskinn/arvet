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
from arvet.core.trial_comparison import TrialComparisonMetric, TrialComparisonResult
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

    @mock.patch('arvet.batch_analysis.tasks.compare_trials_task.autoload_modules')
    def test_load_referenced_models_loads_metric_and_trial_models(self, mock_autoload):
        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
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
        self.assertEqual(mock.call(TrialComparisonMetric, [self.metric.pk]), mock_autoload.call_args_list[0])
        # Order is uncertain for the ids in the second argument, assert separately
        self.assertEqual(TrialResult, mock_autoload.call_args_list[1][0][0])
        self.assertEqual({self.trial_result_1.pk, self.trial_result_2.pk}, set(mock_autoload.call_args_list[1][0][1]))

    def test_result_id_gets_id_without_dereferencing(self):
        result = InitMonitoredResult(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            success=True
        )
        result.save()
        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
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

    @mock.patch('arvet.batch_analysis.tasks.compare_trials_task.autoload_modules')
    def test_get_result_autoloads_model_type_before_dereferencing(self, mock_autoload):
        # Set up objects
        result = InitMonitoredResult(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            success=True
        )
        result.save()

        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
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
            if model == TrialComparisonResult:
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
        self.assertEqual(mock.call(TrialComparisonResult, [result.pk]), mock_autoload.call_args)
        self.assertTrue(constructed)
        self.assertTrue(loaded)
        self.assertTrue(loaded_first)

        # Clean up
        InitMonitoredResult.side_effect = None


class InitMonitoredResult(TrialComparisonResult):
    side_effect = None

    def __init__(self, *args, **kwargs):
        super(InitMonitoredResult, self).__init__(*args, **kwargs)
        if self.side_effect is not None:
            self.side_effect()


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
        comparison_result = mock.create_autospec(TrialComparisonResult)
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

    def test_run_task_saves_result(self):
        comparison_result = mock.create_autospec(TrialComparisonResult)
        self.metric.compare_trials = lambda *args, **kwargs: comparison_result
        subject = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(comparison_result.save.called)
        subject.run_task(self.path_manager)
        self.assertTrue(comparison_result.save.called)

    def test_result_id_is_none_if_result_is_none(self):
        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.UNSTARTED
        )
        self.assertIsNone(obj.result_id)

    def test_result_id_is_result_primary_key(self):
        comparison_result = TrialComparisonResult(
            _id=bson.ObjectId(),
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            success=True
        )
        obj = CompareTrialTask(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            state=JobState.DONE,
            result=comparison_result
        )
        self.assertEqual(comparison_result.pk, obj.result_id)
