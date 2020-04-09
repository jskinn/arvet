# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os
import os.path as path
import bson
import logging
from pathlib import Path
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.core.tests.mock_types as mock_types
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image_source import ImageSource
from arvet.core.image import Image
from arvet.core.image_collection import ImageCollection
from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
import arvet.batch_analysis.tasks.tests.mock_importer as mock_importer


class TestImportDatasetTaskDatabase(unittest.TestCase):
    image = None
    image_collection = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        cls.image = mock_types.make_image()
        cls.image.save()

        cls.image_collection = ImageCollection(
            images=[cls.image],
            timestamps=[1.2],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        cls.image_collection.save()

    def setUp(self):
        # Remove all the tasks at the start of each test
        Task.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageSource._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads_unstarted(self):
        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null',
            state=JobState.UNSTARTED
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_completed(self):
        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null',
            state=JobState.DONE,
            result=self.image_collection
        )
        obj.save()

        # Load all the entities
        all_entities = list(Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_saving_throws_exeption_if_required_fields_are_missing(self):
        obj = ImportDatasetTask(
            # module_name='test.MyTestImporter',
            path='/dev/null',
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            # path='/dev/null',
            state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null'
            # state=JobState.UNSTARTED
        )
        with self.assertRaises(ValidationError):
            obj.save()

    def test_result_id_gets_id_without_dereferencing(self):
        result = InitMonitoredImageSource()
        result.save()
        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null',
            state=JobState.DONE,
            result=result
        )
        obj.save()

        # Set up mocks
        dereferenced = False

        def init_side_effect(_):
            nonlocal dereferenced
            dereferenced = True

        InitMonitoredImageSource.side_effect = init_side_effect

        # Delete and reload the object to reset the references to object ids
        del obj
        obj = next(Task.objects.all())

        # Autoload the model types
        _ = obj.result_id
        self.assertFalse(dereferenced)

        # Clean up
        InitMonitoredImageSource.side_effect = None

    @mock.patch('arvet.batch_analysis.tasks.import_dataset_task.autoload_modules')
    def test_get_result_autoloads_model_type_before_dereferencing(self, mock_autoload):
        # Set up objects
        result = InitMonitoredImageSource()
        result.save()
        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null',
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
            if model == ImageSource:
                loaded = True
        mock_autoload.side_effect = autoload_side_effect

        def init_result_side_effect(_):
            nonlocal loaded, constructed, loaded_first
            constructed = True
            if loaded:
                loaded_first = True
        InitMonitoredImageSource.side_effect = init_result_side_effect

        # Delete and reload the object to reset the references to object ids
        del obj
        obj = next(Task.objects.all())

        # get the result
        obj.get_result()
        self.assertTrue(mock_autoload.called)
        self.assertEqual(mock.call(ImageSource, [result.pk]), mock_autoload.call_args)
        self.assertTrue(constructed)
        self.assertTrue(loaded)
        self.assertTrue(loaded_first)

        # Clean up
        InitMonitoredImageSource.side_effect = None
        result.delete()

    @mock.patch('arvet.batch_analysis.tasks.import_dataset_task.autoload_modules')
    def test_load_referenced_models_autoloads_models_that_are_just_ids(self, mock_autoload):
        # Set up objects
        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null',
            state=JobState.DONE,
            result=self.image_collection
        )
        obj.save()
        obj_id = obj.pk
        del obj     # Clear existing references, which should reset the references to ids

        obj = ImportDatasetTask.objects.get({'_id': obj_id})
        self.assertFalse(mock_autoload.called)
        obj.load_referenced_models()
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(ImageSource, [self.image_collection.pk]), mock_autoload.call_args_list)

    @mock.patch('arvet.batch_analysis.tasks.import_dataset_task.autoload_modules')
    def test_load_referenced_models_does_nothing_to_models_that_are_already_objects(self, mock_autoload):
        # Set up objects
        obj = ImportDatasetTask(
            module_name='test.MyTestImporter',
            path='/dev/null',
            state=JobState.DONE,
            result=self.image_collection
        )
        obj.save()

        self.assertFalse(mock_autoload.called)
        obj.load_referenced_models()
        self.assertFalse(mock_autoload.called)


class InitMonitoredImageSource(mock_types.MockImageSource):
    side_effect = None

    def __init__(self, *args, **kwargs):
        super(InitMonitoredImageSource, self).__init__(*args, **kwargs)
        if self.side_effect is not None:
            self.side_effect()


class TestImportDatasetTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        self.path_manager = PathManager(['~'], '~/tmp')
        mock_importer.reset()
        image = mock_types.make_image()
        self.image_collection = ImageCollection(
            images=[image],
            timestamps=[1.2],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )

    def test_run_task_fails_if_unable_to_import_module(self):
        subject = ImportDatasetTask(
            module_name='fake.not_a_module',
            path=path.abspath(__file__),
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertIsNone(subject.result)
        with self.assertRaises(ImportError):
            subject.run_task(self.path_manager)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_run_task_fails_if_module_doesnt_have_required_method(self):
        subject = ImportDatasetTask(
            module_name='arvet.core.image_source',
            path=path.abspath(__file__),
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_run_task_fails_if_path_doesnt_exist(self):
        subject = ImportDatasetTask(
            module_name=mock_importer.__name__,
            path='not_a_real_path',
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_run_task_fails_if_import_returns_none(self):
        dataset_path = path.abspath(__file__)
        additional_args = {'foo': 'baz'}
        subject = ImportDatasetTask(
            module_name=mock_importer.__name__,
            path=dataset_path,
            additional_args=additional_args,
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertIsNone(subject.result)
        subject.run_task(self.path_manager)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_run_task_fails_if_import_raises_exception(self):
        mock_importer.raise_exception = True
        dataset_path = path.abspath(__file__)
        additional_args = {'foo': 'baz'}
        subject = ImportDatasetTask(
            module_name=mock_importer.__name__,
            path=dataset_path,
            additional_args=additional_args,
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertIsNone(subject.result)
        with self.assertRaises(ValueError):
            subject.run_task(self.path_manager)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_run_task_calls_module_import_dataset_and_stores_result(self):
        mock_importer.return_value = self.image_collection
        dataset_path = path.abspath(__file__)
        additional_args = {'foo': 'baz'}
        subject = ImportDatasetTask(
            module_name=mock_importer.__name__,
            path=dataset_path,
            additional_args=additional_args,
            state=JobState.RUNNING,
            node_id='test',
            job_id=1
        )
        self.assertFalse(mock_importer.called)
        subject.run_task(self.path_manager)
        self.assertTrue(mock_importer.called)
        self.assertEqual(Path(dataset_path), mock_importer.called_path)
        self.assertEqual(additional_args, mock_importer.called_kwargs)

        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        self.assertEqual(self.image_collection, subject.result)

    def test_result_id_is_none_if_result_is_none(self):
        obj = ImportDatasetTask(
            module_name=mock_importer.__name__,
            path='/dev/null',
            state=JobState.UNSTARTED
        )
        self.assertIsNone(obj.result_id)

    def test_result_id_is_result_primary_key(self):
        result = mock_types.MockImageSource()
        result.pk = bson.ObjectId()
        obj = ImportDatasetTask(
            module_name=mock_importer.__name__,
            path='/dev/null',
            state=JobState.DONE,
            result=result
        )
        self.assertEqual(result.pk, obj.result_id)
