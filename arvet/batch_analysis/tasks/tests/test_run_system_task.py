# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import logging
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet.core.image_source import ImageSource
from arvet.core.system import VisionSystem
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


class TestRunSystemTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        self.system = mock_types.MockSystem()
        self.image_source = mock_types.MockImageSource()
        self.path_manager = PathManager(['~'])

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

    def test_run_task_records_returned_metric_result(self):
        trial_result = mock_types.MockTrialResult(system=self.system, image_source=self.image_source, success=True)
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


class TestRunSystemWithSource(unittest.TestCase):

    def setUp(self):
        self._system = mock.create_autospec(VisionSystem)
        self._system.is_image_source_appropriate.return_value = True
        self._system.finish_trial.side_effect = lambda: mock_types.MockTrialResult(system=self._system, success=True)

        self._image_source = mock.create_autospec(ImageSource)
        self._image_source.right_camera_pose = None
        self._image_source.camera_intrinsics = mock.Mock()

        def source_iter():
            for idx in range(10):
                yield idx, mock.Mock()

        self._image_source.__iter__.side_effect = source_iter

    def test_run_system_calls_trial_functions_in_order_mono(self):
        run_system_with_source(self._system, self._image_source)
        mock_calls = self._system.mock_calls
        # set_camera_intrinsics; begin; 10 process image calls; end
        self.assertEqual(13, len(mock_calls))
        self.assertEqual('set_camera_intrinsics', mock_calls[0][0])
        self.assertEqual('start_trial', mock_calls[1][0])
        for i in range(10):
            self.assertEqual('process_image', mock_calls[2 + i][0])
        self.assertEqual('finish_trial', mock_calls[12][0])

    def test_run_system_calls_trial_functions_in_order_stereo(self):
        self._image_source.right_camera_pose = mock.Mock()
        run_system_with_source(self._system, self._image_source)
        mock_calls = self._system.mock_calls
        # set_camera_intrinsics; set_stereo_offset; begin; 10 process image calls; end
        self.assertEqual(14, len(mock_calls))
        self.assertEqual('set_camera_intrinsics', mock_calls[0][0])
        self.assertEqual('set_stereo_offset', mock_calls[1][0])
        self.assertEqual('start_trial', mock_calls[2][0])
        for i in range(10):
            self.assertEqual('process_image', mock_calls[3 + i][0])
        self.assertEqual('finish_trial', mock_calls[13][0])

    def test_run_system_calls_iter(self):
        run_system_with_source(self._system, self._image_source)
        mock_calls = self._image_source.mock_calls
        self.assertEqual(1, len(mock_calls))
        self.assertEqual('__iter__', mock_calls[0][0])

    def test_run_system_returns_trial_result(self):
        result = run_system_with_source(self._system, self._image_source)
        self.assertEqual(result.system, self._system)
        self.assertEqual(result.image_source, self._image_source)
        self.assertTrue(result.success)


class TestRunSystemWithSourceDatabase(unittest.TestCase):
    system = None
    image_source = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
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
        result = run_system_with_source(self.system, self.image_source)
        self.assertEqual(result.system, self.system)
        self.assertEqual(result.image_source, self.image_source)
        self.assertTrue(result.success)
        result.save()
