import unittest
import unittest.mock as mock
import logging
from bson import ObjectId
import pymodm.fields as fields

import arvet.database.tests.database_connection as dbconn
from arvet.batch_analysis.task import Task, JobState
from arvet.config.path_manager import PathManager
import arvet.batch_analysis.scripts.run_task as run_task


class ExpcetedException(Exception):
    pass


class MockTask(Task):
    raise_exception = fields.BooleanField(default=False)
    should_fail = fields.BooleanField(default=False)

    def run_task(self, path_manager: PathManager) -> None:
        if self.raise_exception:
            raise ExpcetedException("Expected Exception")
        if self.should_fail:
            self.mark_job_failed()
        self.mark_job_complete()


class TestRunTask(unittest.TestCase):

    @mock.patch("arvet.batch_analysis.scripts.run_task.autoload_modules", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.Task.objects.get", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_connects_to_database(self, mock_conf_load, mock_dbconfigure, *_):
        db_config = {'test': 22}
        mock_conf_load.return_value = {'paths': ['~'], 'database': db_config, 'image_manager': {'path': '/dev/null'}}

        run_task.main(str(ObjectId()))
        self.assertTrue(mock_dbconfigure.called)
        self.assertEqual(mock.call(db_config), mock_dbconfigure.call_args)

    @mock.patch("arvet.batch_analysis.scripts.run_task.autoload_modules", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.Task.objects.get", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.im_manager.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_configures_image_manager(self, mock_conf_load, _, mock_im_manager, *__):
        im_manager_conf = {'path': '~/my.hdf5'}
        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': im_manager_conf}

        run_task.main(str(ObjectId()))
        self.assertTrue(mock_im_manager.called)
        self.assertEqual(mock.call(im_manager_conf), mock_im_manager.call_args)

    @mock.patch("arvet.batch_analysis.scripts.run_task.Task.objects.get", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.im_manager.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.autoload_modules", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_autoloads_task_type_for_given_task_only(self, mock_conf_load, mock_autoload, *_):
        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': {'path': '/dev/null'}}

        task_id = ObjectId()
        run_task.main(str(task_id))
        self.assertTrue(mock_autoload.called)
        self.assertEqual(mock.call(Task, [task_id]), mock_autoload.call_args)

    @mock.patch("arvet.batch_analysis.scripts.run_task.autoload_modules", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_runs_task(self, mock_conf_load, *_):
        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': {'path': '/dev/null'}}

        mock_task = mock.create_autospec(Task, spec_set=True)
        with mock.patch("arvet.batch_analysis.scripts.run_task.Task.objects.get", autospec=True) as patch_get:
            patch_get.return_value = mock_task
            run_task.main(str(ObjectId()))

        self.assertTrue(mock_task.run_task.called)

    @mock.patch("arvet.batch_analysis.scripts.run_task.autoload_modules", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_marks_job_failed_if_it_raises_exception(self, mock_conf_load, *_):
        logging.disable(logging.CRITICAL)
        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': {'path': '/dev/null'}}

        mock_task = mock.create_autospec(Task, spec_set=True)
        mock_task.run_task.side_effect = ExpcetedException
        with mock.patch("arvet.batch_analysis.scripts.run_task.Task.objects.get", autospec=True) as patch_get:
            patch_get.return_value = mock_task
            with self.assertRaises(ExpcetedException):
                run_task.main(str(ObjectId()))

        self.assertTrue(mock_task.mark_job_failed.called)
        self.assertTrue(mock_task.save.called)
        logging.disable(logging.NOTSET)


class TestRunTaskDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        logging.disable(logging.CRITICAL)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        logging.disable(logging.NOTSET)

    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_raises_exception_if_task_doesnt_exist(self, mock_conf_load, _):
        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': {'path': '/dev/null'}}
        with self.assertRaises(Task.DoesNotExist):
            run_task.main(str(ObjectId()))

    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_runs_task(self, mock_conf_load, _):
        # Make the task
        task = MockTask(state=JobState.RUNNING)
        task.save()

        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': {'path': '/dev/null'}}
        run_task.main(str(task.identifier))

        task.refresh_from_db()
        self.assertTrue(task.is_finished)
        task.delete()

    @mock.patch("arvet.batch_analysis.scripts.run_task.dbconn.configure", autospec=True)
    @mock.patch("arvet.batch_analysis.scripts.run_task.load_global_config", autospec=True)
    def test_marks_job_failed_if_it_raises_exception(self, mock_conf_load, _):
        # Make the task
        task = MockTask(state=JobState.RUNNING, raise_exception=True)
        task.save()

        mock_conf_load.return_value = {'paths': ['~'], 'database': {}, 'image_manager': {'path': '/dev/null'}}
        with self.assertRaises(ExpcetedException):
            run_task.main(str(task.identifier))

        task.refresh_from_db()
        self.assertFalse(task.is_finished)
        self.assertTrue(task.is_unstarted)
        task.delete()
