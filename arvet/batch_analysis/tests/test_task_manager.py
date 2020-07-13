# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import bson
import pymodm.manager
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.system import StochasticBehaviour
import arvet.core.image as im
import arvet.core.image_collection as ic
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
        dbconn.setup_image_manager()

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
        dbconn.tear_down_image_manager()

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
        self.assertEqual(result.result_id, task.result_id)

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


class SeededMockSystem(mock_types.MockSystem):

    @classmethod
    def is_deterministic(cls):
        return StochasticBehaviour.SEEDED


class NonDeterministicMockSystem(mock_types.MockSystem):

    @classmethod
    def is_deterministic(cls):
        return StochasticBehaviour.NON_DETERMINISTIC


class TestTaskManagerRunSystem(unittest.TestCase):
    system = None
    seeded_system = None
    non_deterministic_system = None
    image_source = None
    trial_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.seeded_system = SeededMockSystem()
        cls.non_deterministic_system = NonDeterministicMockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.system.save()
        cls.seeded_system.save()
        cls.non_deterministic_system.save()
        cls.image_source.save()

        cls.trial_result = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

    def setUp(self):
        # Remove all the tasks at the start of the test, so that we're sure it's empty
        Task.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_get_run_system_task_checks_for_existing_task_deterministic(self):
        tmp_manager = RunSystemTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        RunSystemTask.objects = mock_manager
        task_manager.get_run_system_task(self.system, self.image_source, 12, seed=43)
        RunSystemTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('system', query)
        self.assertEqual(self.system._id, query['system'])
        self.assertIn('image_source', query)
        self.assertEqual(self.image_source._id, query['image_source'])
        self.assertNotIn('repeat', query)
        self.assertNotIn('seed', query)

    def test_get_run_system_task_checks_for_existing_task_seeded(self):
        tmp_manager = RunSystemTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        RunSystemTask.objects = mock_manager
        task_manager.get_run_system_task(self.seeded_system, self.image_source, 12, seed=43)
        RunSystemTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('system', query)
        self.assertEqual(self.seeded_system._id, query['system'])
        self.assertIn('image_source', query)
        self.assertEqual(self.image_source._id, query['image_source'])
        self.assertIn('repeat', query)
        self.assertEqual(12, query['repeat'])
        self.assertIn('seed', query)
        self.assertEqual(43, query['seed'])

    def test_get_run_system_task_checks_for_existing_task_non_deterministic(self):
        tmp_manager = RunSystemTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        RunSystemTask.objects = mock_manager
        task_manager.get_run_system_task(self.non_deterministic_system, self.image_source, 12, seed=43)
        RunSystemTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('system', query)
        self.assertEqual(self.non_deterministic_system._id, query['system'])
        self.assertIn('image_source', query)
        self.assertEqual(self.image_source._id, query['image_source'])
        self.assertIn('repeat', query)
        self.assertEqual(12, query['repeat'])
        self.assertNotIn('seed', query)

    def test_get_run_system_task_returns_existing_deterministic(self):
        task = RunSystemTask(
            system=self.system,
            image_source=self.image_source,
            repeat=10,      # Should be ignored
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        result = task_manager.get_run_system_task(self.system, self.image_source, 13)
        self.assertEqual(result.system, task.system)
        self.assertEqual(result.image_source, task.image_source)
        self.assertEqual(result.repeat, 10)
        self.assertIsNone(result.seed)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result_id, task.result_id)

    def test_get_run_system_task_returns_existing_seeded(self):
        task = RunSystemTask(
            system=self.seeded_system,
            image_source=self.image_source,
            repeat=13,
            seed=33,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        result = task_manager.get_run_system_task(self.seeded_system, self.image_source, 13, seed=33)
        self.assertEqual(result.system, task.system)
        self.assertEqual(result.image_source, task.image_source)
        self.assertEqual(result.repeat, task.repeat)
        self.assertEqual(result.seed, task.seed)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result_id, task.result_id)

    def test_get_run_system_task_returns_existing_non_deterministic(self):
        task = RunSystemTask(
            system=self.non_deterministic_system,
            image_source=self.image_source,
            repeat=13,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        result = task_manager.get_run_system_task(self.non_deterministic_system, self.image_source, 13)
        self.assertEqual(result.system, task.system)
        self.assertEqual(result.image_source, task.image_source)
        self.assertEqual(result.repeat, task.repeat)
        self.assertIsNone(result.seed)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result_id, task.result_id)

    def test_get_run_system_task_returns_new_instance_if_no_existing(self):
        repeat = 14
        seed = 121515
        num_cpus = 12
        num_gpus = 3
        for system in [self.system, self.seeded_system, self.non_deterministic_system]:
            task = task_manager.get_run_system_task(
                system=system,
                image_source=self.image_source,
                repeat=repeat,
                seed=seed,
                num_cpus=num_cpus,
                num_gpus=num_gpus
            )
            self.assertIsInstance(task, RunSystemTask)
            self.assertIsNone(task._id)
            self.assertEqual(system, task.system)
            self.assertEqual(self.image_source, task.image_source)
            self.assertEqual(num_cpus, task.num_cpus)
            self.assertEqual(num_gpus, task.num_gpus)
            self.assertTrue(task.is_unstarted)
            self.assertFalse(task.is_running)
            self.assertFalse(task.is_finished)

            if system == self.system:
                self.assertEqual(0, task.repeat)
            else:
                self.assertEqual(task.repeat, repeat)
            if system == self.seeded_system:
                self.assertEqual(seed, task.seed)
            else:
                self.assertIsNone(task.seed)

            # Check it saves and loads with no issues
            task.save()
            loaded_task = RunSystemTask.objects.get({'_id': task.identifier})
            self.assertEqual(task, loaded_task)

    def test_get_run_system_task_ignores_repeat_for_deterministic_systems(self):
        task = RunSystemTask(
            system=self.system,  # The default system is DETERMINISTIC
            image_source=self.image_source,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        for repeat in range(10):
            # No matter the repeat, we should keep getting the same object
            result = task_manager.get_run_system_task(self.system, self.image_source, repeat=repeat)
            self.assertEqual(result.pk, task.pk)
            self.assertEqual(result.system, task.system)
            self.assertEqual(result.image_source, task.image_source)
            self.assertEqual(result.repeat, task.repeat)
            self.assertIsNone(result.seed)
            self.assertEqual(result.num_cpus, 15)
            self.assertEqual(result.num_gpus, 6)
            self.assertEqual(result.state, task.state)
            self.assertEqual(result.result_id, task.result_id)

    def test_get_run_system_task_ignores_seed_for_non_seeded_systems(self):
        task = RunSystemTask(
            system=self.system,     # The default system is DETERMINISTIC
            image_source=self.image_source,
            repeat=13,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        for idx in range(10):
            # No matter the seed, we should keep getting the same object
            seed = (1241151 * idx * idx) % (2 ** 32)
            result = task_manager.get_run_system_task(self.system, self.image_source, 13, seed=seed)
            self.assertEqual(result.pk, task.pk)
            self.assertEqual(result.system, task.system)
            self.assertEqual(result.image_source, task.image_source)
            self.assertEqual(result.repeat, task.repeat)
            self.assertIsNone(result.seed)
            self.assertEqual(result.num_cpus, 15)
            self.assertEqual(result.num_gpus, 6)
            self.assertEqual(result.state, task.state)
            self.assertEqual(result.result_id, task.result_id)

        task = RunSystemTask(
            system=self.non_deterministic_system,
            image_source=self.image_source,
            repeat=22,
            num_cpus=8,
            num_gpus=28,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        for idx in range(10):
            # No matter the seed, we should keep getting the same object
            seed = (1241151 * idx * idx) % (2 ** 32)
            result = task_manager.get_run_system_task(self.non_deterministic_system, self.image_source, 22, seed=seed)
            self.assertEqual(result.pk, task.pk)
            self.assertEqual(result.system, self.non_deterministic_system)
            self.assertEqual(result.image_source, task.image_source)
            self.assertEqual(result.repeat, task.repeat)
            self.assertIsNone(result.seed)
            self.assertEqual(result.num_cpus, 8)
            self.assertEqual(result.num_gpus, 28)
            self.assertEqual(result.state, task.state)
            self.assertEqual(result.result_id, task.result_id)

    def test_get_run_system_task_returns_different_tasks_for_seeded_systems_with_different_seeds(self):
        task = RunSystemTask(
            system=self.seeded_system,
            image_source=self.image_source,
            repeat=13,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        for seed in range(10):
            result = task_manager.get_run_system_task(self.seeded_system, self.image_source, 13, seed=seed)
            self.assertIsNone(result.pk)
            self.assertEqual(result.system, self.seeded_system)
            self.assertEqual(result.image_source, self.image_source)
            self.assertEqual(result.seed, seed)
            self.assertEqual(result.repeat, 13)
            self.assertEqual(result.num_cpus, 1)
            self.assertEqual(result.num_gpus, 0)
            self.assertEqual(result.state, JobState.UNSTARTED)

    def test_get_run_system_task_returns_same_task_for_same_seed(self):
        task = RunSystemTask(
            system=self.seeded_system,
            image_source=self.image_source,
            repeat=13,
            seed=13265,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        result = task_manager.get_run_system_task(self.seeded_system, self.image_source, 13, seed=13265)
        self.assertEqual(result.pk, task.pk)
        self.assertEqual(result.system, task.system)
        self.assertEqual(result.image_source, task.image_source)
        self.assertEqual(result.seed, 13265)
        self.assertEqual(result.repeat, task.repeat)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result_id, task.result_id)

    def test_get_run_system_task_works_with_ids_only(self):
        for system in [self.system, self.seeded_system, self.non_deterministic_system]:
            task = task_manager.get_run_system_task(
                system=system.identifier,
                image_source=self.image_source.identifier,
                repeat=6,
                num_cpus=43,
                num_gpus=89
            )

            # Check it saves and loads with no issues
            task.save()
            loaded_task = RunSystemTask.objects.get({'_id': task.identifier})
            self.assertEqual(task, loaded_task)

    def test_get_run_system_task_determines_stochastic_behaviour_with_only_ids(self):
        task = RunSystemTask(
            system=self.system,     # Default system is DETERMINISTIC
            image_source=self.image_source,
            repeat=6,
            num_cpus=43,
            num_gpus=89,
            state=JobState.UNSTARTED
        )
        task.save()

        for seed in range(10):
            result = task_manager.get_run_system_task(self.system.identifier, self.image_source.identifier,
                                                      repeat=6, seed=seed)
            self.assertEqual(result, task)

        task = RunSystemTask(
            system=self.seeded_system,
            image_source=self.image_source,
            repeat=13,
            seed=13265,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        result = task_manager.get_run_system_task(self.seeded_system.identifier, self.image_source.identifier,
                                                  repeat=13, seed=13265)
        self.assertEqual(result, task)
        for seed in range(10):
            result = task_manager.get_run_system_task(self.seeded_system.identifier, self.image_source.identifier,
                                                      repeat=13, seed=seed)
            self.assertNotEqual(result, task)
            self.assertIsNone(result.pk)

        task = RunSystemTask(
            system=self.non_deterministic_system,
            image_source=self.image_source,
            repeat=22,
            seed=13265,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.trial_result
        )
        task.save()

        for seed in range(10):
            result = task_manager.get_run_system_task(self.non_deterministic_system, self.image_source.identifier,
                                                      repeat=22, seed=seed)
            self.assertEqual(result, task)

    def test_get_run_system_task_cannot_save_with_invalid_ids(self):
        with self.assertRaises(ValueError) as exp:
            task_manager.get_run_system_task(
                system=bson.ObjectId(),
                image_source=self.image_source.identifier,
                repeat=32
            )
        self.assertIn('system', str(exp.exception))

        with self.assertRaises(ValueError) as exp:
            task_manager.get_run_system_task(
                system=self.system,
                image_source=bson.ObjectId(),
                repeat=32
            )
        self.assertIn('image_source', str(exp.exception))


class TestTaskManagerMeasureTrials(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    trial_result_1 = None
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

        cls.trial_result_1 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_1.save()

        cls.trial_result_2 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2.save()

        cls.metric_result = mock_types.MockMetricResult(metric=cls.metric, trial_results=[cls.trial_result_1],
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

    def test_get_measure_trial_task_checks_for_existing_task(self):
        tmp_manager = MeasureTrialTask.objects
        mock_manager = mock.create_autospec(pymodm.manager.Manager)
        MeasureTrialTask.objects = mock_manager
        task_manager.get_measure_trial_task([self.trial_result_1], self.metric)
        MeasureTrialTask.objects = tmp_manager

        self.assertTrue(mock_manager.get.called)
        query = mock_manager.get.call_args[0][0]
        self.assertIn('trial_results', query)
        self.assertEqual({'$all': [self.trial_result_1._id]}, query['trial_results'])
        self.assertIn('metric', query)
        self.assertEqual(self.metric._id, query['metric'])

    def test_get_measure_trial_task_returns_existing(self):
        task = MeasureTrialTask(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.metric_result
        )
        task.save()

        result = task_manager.get_measure_trial_task([self.trial_result_1, self.trial_result_2], self.metric)
        self.assertEqual(result.trial_results, task.trial_results)
        self.assertEqual(result.metric, task.metric)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result_id, task.result_id)

    def test_get_measure_trial_task_returns_existing_changed_order(self):
        task = MeasureTrialTask(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric,
            num_cpus=15,
            num_gpus=6,
            state=JobState.DONE,
            result=self.metric_result
        )
        task.save()

        result = task_manager.get_measure_trial_task([self.trial_result_2, self.trial_result_1], self.metric)
        self.assertEqual(result.trial_results, task.trial_results)
        self.assertEqual(result.metric, task.metric)
        self.assertEqual(result.num_cpus, 15)
        self.assertEqual(result.num_gpus, 6)
        self.assertEqual(result.state, task.state)
        self.assertEqual(result.result_id, task.result_id)

    def test_get_measure_trial_task_returns_new_instance_if_no_existing(self):
        num_cpus = 12
        num_gpus = 3
        result = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric,
            num_cpus=num_cpus,
            num_gpus=num_gpus
        )
        self.assertIsInstance(result, MeasureTrialTask)
        self.assertIsNone(result._id)
        self.assertEqual(result.trial_results, [self.trial_result_1, self.trial_result_2])
        self.assertEqual(result.metric, self.metric)
        self.assertEqual(result.num_cpus, num_cpus)
        self.assertEqual(result.num_gpus, num_gpus)
        self.assertTrue(result.is_unstarted)
        self.assertFalse(result.is_running)
        self.assertFalse(result.is_finished)

    def test_get_measure_trial_task_works_with_ids_only(self):
        task = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result_1.identifier],
            metric=self.metric.identifier,
            num_cpus=32,
            num_gpus=7
        )

        # Check it saves and loads with no issues
        task.save()
        loaded_task = MeasureTrialTask.objects.get({'_id': task.identifier})
        self.assertEqual(task, loaded_task)

    def test_get_measure_trial_task_cannot_save_with_invalid_ids(self):
        with self.assertRaises(ValueError) as exp:
            task_manager.get_measure_trial_task(
                trial_results=[self.trial_result_1, bson.ObjectId()],
                metric=self.metric
            )
        self.assertIn('trial_result', str(exp.exception))

        with self.assertRaises(ValueError) as exp:
            task_manager.get_measure_trial_task(
                trial_results=[self.trial_result_1.identifier],
                metric=bson.ObjectId()
            )
        self.assertIn('metric', str(exp.exception))


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

        cls.trial_result_1 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
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
        mock_types.MockTrialResult._mongometa.collection.drop()
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
        self.assertEqual(result.result_id, task.result_id)

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

    def test_get_compare_trial_task_works_with_ids_only(self):
        task = task_manager.get_trial_comparison_task(
            trial_results_1=[self.trial_result_1.identifier],
            trial_results_2=[self.trial_result_2.identifier],
            comparison_metric=self.metric.identifier,
            num_cpus=907,
            num_gpus=67
        )

        # Check it saves and loads with no issues
        task.save()
        loaded_task = CompareTrialTask.objects.get({'_id': task.identifier})
        self.assertEqual(task, loaded_task)

    def test_get_compare_trial_task_cannot_save_with_invalid_ids(self):
        with self.assertRaises(ValueError) as exp:
            task_manager.get_trial_comparison_task(
                trial_results_1=[self.trial_result_1, bson.ObjectId()],
                trial_results_2=[self.trial_result_2.identifier],
                comparison_metric=self.metric.identifier
            )
        self.assertIn('trial_results_1', str(exp.exception))

        with self.assertRaises(ValueError) as exp:
            task_manager.get_trial_comparison_task(
                trial_results_1=[self.trial_result_1.identifier],
                trial_results_2=[self.trial_result_2, bson.ObjectId()],
                comparison_metric=self.metric.identifier
            )
        self.assertIn('trial_results_2', str(exp.exception))

        with self.assertRaises(ValueError) as exp:
            task_manager.get_trial_comparison_task(
                trial_results_1=[self.trial_result_1.identifier],
                trial_results_2=[self.trial_result_2.identifier],
                comparison_metric=bson.ObjectId()
            )
        self.assertIn('metric', str(exp.exception))


class TestTaskManagerScheduleTasksDatabase(unittest.TestCase):
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

        cls.trial_result_1 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_1.save()
        cls.trial_result_2.save()

        cls.metric_result = mock_types.MockMetricResult(metric=cls.metric, trial_results=[cls.trial_result_1],
                                                        success=True)
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
        mock_types.MockTrialResult._mongometa.collection.drop()
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

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

    def test_schedule_tasks_schedules_run_system_task(self):
        task = task_manager.get_run_system_task(
            image_source=self.image_source,
            system=self.system
        )
        task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

    def test_schedule_tasks_schedules_measure_trial_task(self):
        task = task_manager.get_measure_trial_task(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric
        )
        task.save()

        mock_job_system = mock.create_autospec(spec=arvet.batch_analysis.job_system.JobSystem, spec_set=True)
        mock_job_system.node_id = 'here'
        mock_job_system.is_job_running.return_value = False

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

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

        task_manager.schedule_tasks(mock_job_system)

        self.assertEqual(mock.call(task), mock_job_system.run_task.call_args)

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

        self.assertEqual(1, mock_job_system.run_task.call_count)
        self.assertEqual(mock.call(included_task), mock_job_system.run_task.call_args)
        self.assertNotIn(mock.call(excluded_task), mock_job_system.run_task.call_args_list)


class TestTaskManagerCountPendingTasks(unittest.TestCase):
    system = None
    image = None
    image_collection = None
    metric = None
    comparison_metric = None
    trial_result_1 = None
    trial_result_2 = None
    metric_result = None
    comaprison_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

        cls.image = mock_types.make_image()
        cls.image.save()

        cls.image_collection = ic.ImageCollection(
            images=[cls.image],
            timestamps=[1.2],
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        cls.image_collection.save()

        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.metric = mock_types.MockMetric()
        cls.comparison_metric = mock_types.MockTrialComparisonMetric()
        cls.system.save()
        cls.metric.save()
        cls.comparison_metric.save()

        cls.trial_result_1 = mock_types.MockTrialResult(
            image_source=cls.image_collection, system=cls.system, success=True)
        cls.trial_result_2 = mock_types.MockTrialResult(
            image_source=cls.image_collection, system=cls.system, success=True)
        cls.trial_result_1.save()
        cls.trial_result_2.save()

        cls.metric_result = mock_types.MockMetricResult(metric=cls.metric, trial_results=[cls.trial_result_1],
                                                        success=True)
        cls.metric_result.save()
        cls.comparison_result = tcmp.TrialComparisonResult(
            metric=cls.comparison_metric,
            trial_results_1=[cls.trial_result_1],
            trial_results_2=[cls.trial_result_2],
            success=True)
        cls.comparison_result.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        tcmp.TrialComparisonResult._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        ic.ImageCollection._mongometa.collection.drop()
        im.Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_returns_number_of_incomplete_tasks(self):
        # Set up initial database state. We have 3 tasks of each type, 1 complete, 1 running, 1 unstarted of each
        Task._mongometa.collection.drop()
        # Import dataset tasks
        task = ImportDatasetTask(
            module_name='test',
            path='/dev/null',
            state=JobState.UNSTARTED
        )
        task.save()
        task = ImportDatasetTask(
            module_name='test',
            path='/dev/null',
            state=JobState.RUNNING,
            node_id='here',
            job_id=10
        )
        task.save()
        task = ImportDatasetTask(
            module_name='my_importer',
            path='/dev/null',
            state=JobState.DONE,
            result=self.image_collection
        )
        task.save()

        # Run system tasks
        task = RunSystemTask(
            system=self.system,
            image_source=self.image_collection,
            repeat=1,
            state=JobState.UNSTARTED
        )
        task.save()
        task = RunSystemTask(
            system=self.system,
            image_source=self.image_collection,
            repeat=2,
            state=JobState.RUNNING,
            node_id='here',
            job_id=11
        )
        task.save()
        task = RunSystemTask(
            system=self.system,
            image_source=self.image_collection,
            repeat=3,
            state=JobState.DONE,
            result=self.trial_result_1
        )
        task.save()

        # Measure trial tasks
        task = MeasureTrialTask(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric,
            state=JobState.UNSTARTED
        )
        task.save()
        task = MeasureTrialTask(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric,
            state=JobState.RUNNING,
            node_id='here',
            job_id=12
        )
        task.save()
        task = MeasureTrialTask(
            trial_results=[self.trial_result_1, self.trial_result_2],
            metric=self.metric,
            state=JobState.DONE,
            result=self.metric_result
        )
        task.save()

        # Compare trials tasks
        task = CompareTrialTask(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            metric=self.comparison_metric,
            state=JobState.UNSTARTED
        )
        task.save()
        task = CompareTrialTask(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            metric=self.comparison_metric,
            state=JobState.RUNNING,
            node_id='here',
            job_id=12
        )
        task.save()
        task = CompareTrialTask(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            metric=self.comparison_metric,
            state=JobState.DONE,
            result=self.comparison_result
        )
        task.save()

        # Tell me how many are not done
        # Should be 12 total, 4 of which are done, leaving 8
        self.assertEqual(8, task_manager.count_pending_tasks())
