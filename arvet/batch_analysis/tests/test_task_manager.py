# Copyright (c) 2017, John Skinner
import os
import unittest
import unittest.mock as mock
import pymodm.manager
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.core.tests.mock_types as mock_types
from arvet.core.sequence_type import ImageSequenceType
import arvet.core.image as im
import arvet.core.image_collection as ic
import arvet.core.trial_result as tr
import arvet.core.metric as mtr
import arvet.core.trial_comparison as tcmp
import arvet.batch_analysis.task_manager as task_manager
import arvet.batch_analysis.job_system

from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask
from arvet.batch_analysis.tasks.compare_trials_task import CompareTrialTask


class TestTaskManagerImportDataset(unittest.TestCase):
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

    def test_get_import_dataset_task_checks_for_existing_task(self):
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        additional_args = {'foo': 'bar'}
        tmp_manager = ImportDatasetTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        ImportDatasetTask.objects = mock_manager
        task_manager.get_import_dataset_task(module_name, path, additional_args)
        ImportDatasetTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('module_name', query)
        self.assertEqual(module_name, query['module_name'])
        self.assertIn('path', query)
        self.assertEqual(path, query['path'])
        self.assertIn('additional_args', query)
        self.assertEqual(additional_args, query['additional_args'])

    def test_get_import_dataset_task_returns_existing(self):
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        additional_args = {'foo': 'bar'}
        task = ImportDatasetTask(
            module_name=module_name,
            path=path,
            additional_args=additional_args,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.image_collection
        )
        task.save()

        result = task_manager.get_import_dataset_task(module_name, path, additional_args)
        self.assertEqual(result.module_name, task.module_name)
        self.assertEqual(result.path, task.path)
        self.assertEqual(result.additional_args, task.additional_args)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result, task.result)

    def test_get_import_dataset_task_returns_new_instance_if_no_existing(self):
        module_name = 'test_module'
        path = '/tmp/dataset/thisisadataset'
        additional_args = {}
        num_cpus = 12
        num_gpus = 3
        result = task_manager.get_import_dataset_task(
            module_name=module_name,
            path=path,
            additional_args=additional_args,
            num_cpus=num_cpus,
            num_gpus=num_gpus
        )
        self.assertIsInstance(result, ImportDatasetTask)
        self.assertIsNone(result._id)
        self.assertEqual(result.module_name, module_name)
        self.assertEqual(result.path, path)
        self.assertEqual(result.additional_args, additional_args)
        self.assertEqual(result.num_cpus, num_cpus)
        self.assertEqual(result.num_gpus, num_gpus)
        self.assertTrue(result.is_unstarted)
        self.assertFalse(result.is_running)
        self.assertFalse(result.is_finished)


class TestTaskManagerRunSystem(unittest.TestCase):
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

        cls.trial_result = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        tr.TrialResult._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_get_run_system_task_checks_for_existing_task(self):
        tmp_manager = RunSystemTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        RunSystemTask.objects = mock_manager
        task_manager.get_run_system_task(self.system, self.image_source, 12)
        RunSystemTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('system', query)
        self.assertEqual(self.system._id, query['system'])
        self.assertIn('image_source', query)
        self.assertEqual(self.image_source._id, query['image_source'])
        self.assertIn('repeat', query)
        self.assertEqual(12, query['repeat'])

    def test_get_run_system_task_returns_existing(self):
        task = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            repeat=13,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        result = task_manager.get_run_system_task(self.system, self.image_source, 13)
        self.assertEqual(result.system, task.system)
        self.assertEqual(result.image_source, task.image_source)
        self.assertEqual(result.repeat, task.repeat)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result, task.result)

    def test_get_run_system_task_returns_new_instance_if_no_existing(self):
        repeat = 14
        num_cpus = 12
        num_gpus = 3
        result = task_manager.get_run_system_task(
            system=self.system,
            image_source=self.image_source,
            repeat=repeat,
            num_cpus=num_cpus,
            num_gpus=num_gpus
        )
        self.assertIsInstance(result, RunSystemTask)
        self.assertIsNone(result._id)
        self.assertEqual(result.system, self.system)
        self.assertEqual(result.image_source, self.image_source)
        self.assertEqual(result.repeat, repeat)
        self.assertEqual(result.num_cpus, num_cpus)
        self.assertEqual(result.num_gpus, num_gpus)
        self.assertTrue(result.is_unstarted)
        self.assertFalse(result.is_running)
        self.assertFalse(result.is_finished)


class TestTaskManagerMeasureTrials(unittest.TestCase):
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

        cls.trial_result = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

        cls.metric_result = mtr.MetricResult(metric=cls.metric, trial_results=[cls.trial_result], success=True)
        cls.metric_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        mtr.MetricResult._mongometa.collection.drop()
        tr.TrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_get_measure_trial_task_checks_for_existing_task(self):
        tmp_manager = MeasureTrialTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        MeasureTrialTask.objects = mock_manager
        task_manager.get_measure_trial_task([self.trial_result], self.metric)
        MeasureTrialTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('trial_results', query)
        self.assertEqual([self.trial_result._id], query['trial_results'])
        self.assertIn('metric', query)
        self.assertEqual(self.metric._id, query['metric'])

    def test_get_measure_trial_task_returns_existing(self):
        task = MeasureTrialTask(
            trial_results=[self.trial_result],
            metric=self.metric,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.metric_result
        )
        task.save()

        result = task_manager.get_measure_trial_task([self.trial_result], self.metric)
        self.assertEqual(result.trial_results, task.trial_results)
        self.assertEqual(result.metric, task.metric)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result, task.result)

    def test_get_measure_trial_task_returns_new_instance_if_no_existing(self):
        num_cpus = 12
        num_gpus = 3
        result = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result],
            metric=self.metric,
            num_cpus=num_cpus,
            num_gpus=num_gpus
        )
        self.assertIsInstance(result, MeasureTrialTask)
        self.assertIsNone(result._id)
        self.assertEqual(result.trial_results, [self.trial_result])
        self.assertEqual(result.metric, self.metric)
        self.assertEqual(result.num_cpus, num_cpus)
        self.assertEqual(result.num_gpus, num_gpus)
        self.assertTrue(result.is_unstarted)
        self.assertFalse(result.is_running)
        self.assertFalse(result.is_finished)


class TestTaskManagerCompareTrials(unittest.TestCase):
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

        cls.trial_result_1 = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2 = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_1.save()
        cls.trial_result_2.save()

        cls.metric_result = tcmp.TrialComparisonResult(
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
        tcmp.TrialComparisonResult._mongometa.collection.drop()
        tr.TrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_get_trial_comparison_task_checks_for_existing_task(self):
        tmp_manager = CompareTrialTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        CompareTrialTask.objects = mock_manager
        task_manager.get_trial_comparison_task([self.trial_result_1], [self.trial_result_2], self.metric)
        CompareTrialTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('trial_results_1', query)
        self.assertEqual([self.trial_result_1._id], query['trial_results_1'])
        self.assertIn('trial_results_2', query)
        self.assertEqual([self.trial_result_2._id], query['trial_results_2'])
        self.assertIn('metric', query)
        self.assertEqual(self.metric._id, query['metric'])

    def test_get_trial_comparison_task_returns_existing(self):
        task = CompareTrialTask(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            metric=self.metric,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.metric_result
        )
        task.save()

        result = task_manager.get_trial_comparison_task([self.trial_result_1], [self.trial_result_2], self.metric)
        self.assertEqual(result.trial_results_1, task.trial_results_1)
        self.assertEqual(result.trial_results_2, task.trial_results_2)
        self.assertEqual(result.metric, task.metric)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result, task.result)

    def test_get_trial_comparison_task_returns_new_instance_if_no_existing(self):
        num_cpus = 12
        num_gpus = 3
        result = task_manager.get_trial_comparison_task(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            comparison_metric=self.metric,
            num_cpus=num_cpus,
            num_gpus=num_gpus
        )
        self.assertIsInstance(result, CompareTrialTask)
        self.assertIsNone(result._id)
        self.assertEqual(result.trial_results_1, [self.trial_result_1])
        self.assertEqual(result.trial_results_2, [self.trial_result_2])
        self.assertEqual(result.metric, self.metric)
        self.assertEqual(result.num_cpus, num_cpus)
        self.assertEqual(result.num_gpus, num_gpus)
        self.assertTrue(result.is_unstarted)
        self.assertFalse(result.is_running)
        self.assertFalse(result.is_finished)


class TestTaskManagerScheduleTasks(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    comparison_metric = None
    trial_result_1 = None
    trial_result_2 = None
    metric_result = None
    comaprison_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.metric = mock_types.MockMetric()
        cls.comparison_metric = mock_types.MockTrialComparisonMetric()
        cls.system.save()
        cls.image_source.save()
        cls.metric.save()
        cls.comparison_metric.save()

        cls.trial_result_1 = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2 = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_1.save()
        cls.trial_result_2.save()

        cls.metric_result = mtr.MetricResult(metric=cls.metric, trial_results=[cls.trial_result_1], success=True)
        cls.metric_result.save()
        cls.comparison_result = tcmp.TrialComparisonResult(
            metric=cls.comparison_metric,
            trial_results_1=[cls.trial_result_1],
            trial_results_2=[cls.trial_result_2],
            success=True)
        cls.comparison_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        tcmp.TrialComparisonResult._mongometa.collection.drop()
        tr.TrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_schedule_tasks_cancels_tasks_listed_as_running_on_this_node(self):
        # Set up initial database state
        this_node_task = ImportDatasetTask(
            module_name='test',
            path='/dev/null',
            state=JobState.RUNNING,
            node_id='here',
            job_id=10
        )
        this_node_task.save()
        different_node_task = ImportDatasetTask(
            module_name='test',
            path='/dev/null',
            state=JobState.RUNNING,
            node_id='there',
            job_id=14
        )
        different_node_task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = None    # Ensure no new tasks are successfully run

        task_manager.schedule_tasks(mock_job_system)

        task = ImportDatasetTask.objects.get({'_id': this_node_task._id})
        self.assertEqual(JobState.UNSTARTED, task.state)

        task = ImportDatasetTask.objects.get({'_id': different_node_task._id})
        self.assertEqual(JobState.RUNNING, task.state)

        self.assertTrue(mock_job_system.is_job_running.called)
        self.assertEqual(1, mock_job_system.is_job_running.call_count)
        self.assertEqual(10, mock_job_system.is_job_running.call_args[0][0])

    def test_schedule_tasks_schedules_import_dataset_task(self):
        task = task_manager.get_import_dataset_task(module_name='test', path='/dev/null')
        task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

        task.refresh_from_db()
        self.assertTrue(task.is_running)
        self.assertEqual('here', task.node_id)
        self.assertEqual(1433, task.job_id)

    def test_schedule_tasks_schedules_run_system_task(self):
        task = task_manager.get_run_system_task(
            image_source=self.image_source,
            system=self.system
        )
        task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

        task.refresh_from_db()
        self.assertTrue(task.is_running)
        self.assertEqual('here', task.node_id)
        self.assertEqual(1433, task.job_id)

    def test_schedule_tasks_schedules_measure_trial_task(self):
        task = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric
        )
        task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

        task.refresh_from_db()
        self.assertTrue(task.is_running)
        self.assertEqual('here', task.node_id)
        self.assertEqual(1433, task.job_id)

    def test_schedule_tasks_schedules_compare_trials_task(self):
        task = task_manager.get_trial_comparison_task(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            comparison_metric=self.comparison_metric
        )
        task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 1433

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

        task.refresh_from_db()
        self.assertTrue(task.is_running)
        self.assertEqual('here', task.node_id)
        self.assertEqual(1433, task.job_id)

    def test_schedule_tasks_doesnt_rerun_completed_tasks(self):
        # Set up initial database state
        unfinished_task = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            repeat=1,
            state=JobState.RUNNING,
            node_id='here',
            job_id=10
        )
        unfinished_task.save()
        finished_task = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            repeat=1,
            state=JobState.DONE,
            result=self.trial_result_1
        )
        finished_task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = True
        mock_job_system.run_task.return_value = 33

        task_manager.schedule_tasks(mock_job_system)

        task = Task.objects.get({'_id': unfinished_task._id})
        self.assertEqual(JobState.RUNNING, task.state)

        task = Task.objects.get({'_id': finished_task._id})
        self.assertEqual(JobState.DONE, task.state)

        self.assertTrue(mock_job_system.is_job_running.called)
        self.assertEqual(1, mock_job_system.is_job_running.call_count)
        self.assertEqual(10, mock_job_system.is_job_running.call_args[0][0])
        self.assertFalse(mock_job_system.run_task.called)

    def test_schedule_tasks_can_limit_task_ids(self):
        # Set up initial database state
        included_task = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result_1],
            metric=self.metric
        )
        included_task.save()
        excluded_task = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result_2],
            metric=self.metric
        )
        excluded_task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False
        mock_job_system.run_task.return_value = 33

        task_manager.schedule_tasks(mock_job_system, task_ids=[included_task.identifier])

        task = Task.objects.get({'_id': included_task._id})
        self.assertEqual(JobState.RUNNING, task.state)
        self.assertEqual(33, task.job_id)
        self.assertEqual('here', task.node_id)

        task = Task.objects.get({'_id': excluded_task._id})
        self.assertEqual(JobState.UNSTARTED, task.state)

        self.assertTrue(mock_job_system.run_task.called)
        self.assertEqual(1, mock_job_system.run_task.call_count)
        self.assertEqual(mock.call(included_task), mock_job_system.run_task.call_args)
