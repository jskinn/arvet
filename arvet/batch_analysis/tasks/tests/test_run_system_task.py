# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import logging
import bson
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet.core.image_source import ImageSource
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask, run_system_with_source


class TestRunSystemTaskDatabase(unittest.TestCase):
    system = None
    image_source = None
    trial_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.system.save()
        cls.image_source.save()

        cls.trial_result = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_stores_and_loads_unstarted(self):
        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.UNSTARTED
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_completed(self):
        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.DONE,
            result=self.trial_result
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_saving_throws_exeption_if_required_fields_are_missing(self):
        obj = RunSystemTask(
            # system=self.system,
            image_source=self.image_source,
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = RunSystemTask(
            system=self.system,
            # image_source=self.image_source,
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source
            # state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

    @mock.patch('arvet.batch_analysis.tasks.run_system_task.autoload_modules')
    def test_load_referenced_models_loads_system_and_image_source_models(self, mock_autoload):
        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.UNSTARTED
        )
        obj.save()

        # Delete and reload the object to reset the references to object ids
        del obj
        obj = next(Task.objects.all())

        # Auto load the model types
        self.assertFalse(mock_autoload.called)
        obj.load_referenced_modules()
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(VisionSystem, [self.system.pk]), mock_autoload.call_args_list)
        self.assertIn(mock.call(ImageSource, [self.image_source.pk]), mock_autoload.call_args_list)

    def test_result_id_gets_id_without_dereferencing(self):
        result = InitMonitoredResult(
            system=self.system,
            image_source=self.image_source,
            success=True
        )
        result.save()
        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
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

    @mock.patch('arvet.batch_analysis.tasks.run_system_task.autoload_modules')
    def test_get_result_autoloads_model_type_before_dereferencing(self, mock_autoload):
        # Set up objects
        result = InitMonitoredResult(
            system=self.system,
            image_source=self.image_source,
            success=True
        )
        result.save()

        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
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
            if model == TrialResult:
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
        self.assertEqual(mock.call(TrialResult, [result.pk]), mock_autoload.call_args)
        self.assertTrue(constructed)
        self.assertTrue(loaded)
        self.assertTrue(loaded_first)

        # Clean up
        InitMonitoredResult.side_effect = None


class InitMonitoredResult(mock_types.MockTrialResult):
    side_effect = None

    def __init__(self, *args, **kwargs):
        super(InitMonitoredResult, self).__init__(*args, **kwargs)
        if self.side_effect is not None:
            self.side_effect()


class TestRunSystemTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        self.system = mock_types.MockSystem()
        self.image_source = mock_types.MockImageSource()
        self.path_manager = PathManager(['~'], '~/tmp')

    def test_run_task_records_unable_to_measure_trial(self):
        self.system.is_image_source_appropriate = lambda _: False
        subject = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
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
        self.assertEqual(self.system, subject.result.system)
        self.assertEqual(self.image_source, subject.result.image_source)

    def test_run_task_records_exception_during_execution_and_re_raises(self):
        message = 'No mercy. No respite.'

        def bad_measure_results(*_, **__):
            raise ValueError(message)

        self.system.start_trial = bad_measure_results
        subject = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
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
        self.assertEqual(self.system, subject.result.system)
        self.assertEqual(self.image_source, subject.result.image_source)

    def test_run_task_records_returned_trial_result(self):
        trial_result = mock.create_autospec(mock_types.MockTrialResult)
        self.system.finish_trial = lambda: trial_result
        subject = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_finished)
        self.assertEqual(subject.result, trial_result)

    def test_run_task_saves_result(self):
        trial_result = mock.create_autospec(mock_types.MockTrialResult)
        self.system.finish_trial = lambda: trial_result
        subject = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(trial_result.save.called)
        subject.run_task(self.path_manager)
        self.assertTrue(trial_result.save.called)

    def test_result_id_is_none_if_result_is_none(self):
        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.UNSTARTED
        )
        self.assertIsNone(obj.result_id)

    def test_result_id_is_result_primary_key(self):
        result = mock_types.MockTrialResult()
        result.pk = bson.ObjectId()
        obj = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            state=JobState.DONE,
            result=result
        )
        self.assertEqual(result.pk, obj.result_id)


class TestRunSystemWithSource(unittest.TestCase):
    # These are the system functions we care about what order they are called in
    # systems have other functions (like __str__, is_deterministic)
    # but we don't care if they're called out of order
    stateful_functions = {
        # these first 4 must be called before
        'resolve_paths',
        'set_camera_intrinsics',
        'set_stereo_offset',
        'preload_image_data',
        'start_trial',
        'process_image',
        'finish_trial'
    }

    def setUp(self):
        self.path_manager = PathManager(['~'], '~/tmp')
        self.system = mock.create_autospec(VisionSystem)
        self.system.is_image_source_appropriate.return_value = True
        self.system.finish_trial.side_effect = lambda: mock_types.MockTrialResult(system=self.system, success=True)

        self.image_source = mock.create_autospec(ImageSource)
        self.image_source.stereo_offset = None
        self.image_source.camera_intrinsics = mock.Mock()

        def source_iter():
            for idx in range(10):
                yield idx, mock.Mock()

        self.image_source.__iter__.side_effect = source_iter

    def test_run_system_calls_trial_functions_in_order_mono(self):
        run_system_with_source(self.system, self.image_source, self.path_manager)
        mock_calls = self.system.mock_calls
        system_calls = [mock_call[0] for mock_call in mock_calls if mock_call[0] in self.stateful_functions]

        # set_camera_intrinsics; begin; 10 process image calls; end
        self.assertEqual(24, len(system_calls))
        self.assertEqual('resolve_paths', system_calls[0])
        self.assertEqual('set_camera_intrinsics', system_calls[1])

        # first, preload the images
        for i in range(10):
            self.assertEqual('preload_image_data', system_calls[2 + i])

        # then, do the trial
        self.assertEqual('start_trial', system_calls[12])
        for i in range(10):
            self.assertEqual('process_image', system_calls[13 + i])
        self.assertEqual('finish_trial', system_calls[23])

    def test_run_system_calls_trial_functions_in_order_stereo(self):
        self.image_source.stereo_offset = mock.Mock()
        run_system_with_source(self.system, self.image_source, self.path_manager)
        mock_calls = self.system.mock_calls
        system_calls = [mock_call[0] for mock_call in mock_calls if mock_call[0] in self.stateful_functions]

        # set_camera_intrinsics; set_stereo_offset; begin; 10 process image calls; end
        self.assertEqual(25, len(system_calls))
        self.assertEqual('resolve_paths', system_calls[0])
        self.assertEqual('set_camera_intrinsics', system_calls[1])
        self.assertEqual('set_stereo_offset', system_calls[2])

        # first, preload the images
        for i in range(10):
            self.assertEqual('preload_image_data', system_calls[3 + i])

        # then, do the trial
        self.assertEqual('start_trial', system_calls[13])
        for i in range(10):
            self.assertEqual('process_image', system_calls[14 + i])
        self.assertEqual('finish_trial', system_calls[24])

    def test_run_system_calls_iter(self):
        run_system_with_source(self.system, self.image_source, self.path_manager)
        self.assertTrue(self.image_source.__iter__.called)

    def test_run_system_returns_trial_result(self):
        result = run_system_with_source(self.system, self.image_source, self.path_manager)
        self.assertEqual(result.system, self.system)
        self.assertEqual(result.image_source, self.image_source)
        self.assertTrue(result.success)


class TestRunSystemWithSourceDatabase(unittest.TestCase):
    system = None
    image_source = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.path_manager = PathManager(['~'], '~/tmp')
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.system.save()
        cls.image_source.save()

        # cls.system.is_image_source_appropriate.return_value = True
        # cls.system.finish_trial.side_effect = lambda: tr.TrialResult(system=cls.system, success=True)

        #cls.image_source.right_camera_pose = None
        #cls.image_source.camera_intrinsics = mock.Mock()

        #def source_iter():
        #    for idx in range(10):
        #        yield idx, mock.Mock()

        #cls.image_source.__iter__.side_effect = source_iter

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        mock_types.MockTrialResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_run_system_returns_trial_result_that_can_be_saved(self):
        result = run_system_with_source(self.system, self.image_source, self.path_manager)
        self.assertEqual(result.system, self.system)
        self.assertEqual(result.image_source, self.image_source)
        self.assertTrue(result.success)
        result.save()
