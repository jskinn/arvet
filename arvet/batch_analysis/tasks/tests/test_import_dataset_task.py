# Copyright (c) 2017, John Skinner
import unittest
import os
import os.path as path
import logging
from pymodm.errors import ValidationError
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.core.tests.mock_types as mock_types
from arvet.core.sequence_type import ImageSequenceType
import arvet.core.image as im
import arvet.core.image_collection as ic
from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
import arvet.batch_analysis.tasks.tests.mock_importer as mock_importer


class TestMeasureTrialTaskDatabase(unittest.TestCase):
    image = None
    image_collection = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        cls.image = mock_types.make_image()
        cls.image.save()

        cls.image_collection = ic.ImageCollection(
            images=[cls.image],
            timestamps=[1.2],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        cls.image_collection.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        im.Image._mongometa.collection.drop()
        ic.ImageCollection._mongometa.collection.drop()
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



class TestImportDatasetTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def setUp(self):
        self.path_manager = PathManager(['~'])
        mock_importer.reset()
        image = mock_types.make_image()
        self.image_collection = ic.ImageCollection(
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
        self.assertEqual(dataset_path, mock_importer.called_path)
        self.assertEqual(additional_args, mock_importer.called_kwargs)

        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        self.assertEqual(self.image_collection, subject.result)
