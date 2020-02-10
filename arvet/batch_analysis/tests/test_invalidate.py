import os
import unittest
import unittest.mock as mock

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.core.tests.mock_types as mock_types
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image
from arvet.core.image_source import ImageSource
from arvet.core.image_collection import ImageCollection
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult

from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask

import arvet.batch_analysis.invalidate as invalidate


class TestInvalidateDatasetLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collections for models we used
        Task._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self):
        # Remove the collections as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()

        self.dataset_loaders = ['dataset.loader.loader_A', 'dataset.loader.loader_B']
        self.image_collections = {}
        self.import_dataset_tasks = {}

        for dataset_loader in self.dataset_loaders:
            # Make multiple image collections for each dataset loader

            self.image_collections[dataset_loader] = []
            self.import_dataset_tasks[dataset_loader] = []

            # Add completed import dataset tasks with image collections
            for _ in range(2):
                image_collection = make_image_collection()
                self.image_collections[dataset_loader].append(image_collection.identifier)
                task = ImportDatasetTask(
                    module_name=dataset_loader,
                    path='test/filename',
                    result=image_collection,
                    state=JobState.DONE
                )
                task.save()
                self.import_dataset_tasks[dataset_loader].append(task.identifier)

            # Add incomplete load dataset tasks
            for _ in range(3):
                task = ImportDatasetTask(
                    module_name=dataset_loader,
                    path='test/filename',
                    state=JobState.UNSTARTED
                )
                task.save()
                self.import_dataset_tasks[dataset_loader].append(task.identifier)

    def test_invalidate_dataset_loader_invalidates_image_collections(self):
        num_image_collections = sum(len(group) for group in self.image_collections.values())

        # Check that we start with the right number of tasks and image collections
        self.assertEqual(num_image_collections, ImageCollection.objects.all().count())
        self.assertEqual(num_image_collections + 3 * len(self.dataset_loaders), Task.objects.all().count())

        with mock.patch('arvet.batch_analysis.invalidate.invalidate_image_collections',
                        wraps=invalidate.invalidate_image_collections) as \
                mock_invalidate_image_collection:
            invalidate.invalidate_dataset_loaders([self.dataset_loaders[0]])

            # Check that all the image collections imported by this loader are invalidated
            self.assertEqual(1, len(mock_invalidate_image_collection.call_args_list))
            for image_collection_id in self.image_collections[self.dataset_loaders[0]]:
                self.assertIn(image_collection_id,
                              mock_invalidate_image_collection.call_args_list[0][0][0])

            # Check that the other image collections are not invalidated
            for i in range(1, len(self.dataset_loaders)):
                for image_collection_id in self.image_collections[self.dataset_loaders[i]]:
                    self.assertNotIn(image_collection_id,
                                     mock_invalidate_image_collection.call_args_list[0][0][0])

        reduced_image_collections = sum(len(self.image_collections[self.dataset_loaders[idx]])
                                        for idx in range(1, len(self.dataset_loaders)))

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(reduced_image_collections, ImageCollection.objects.all().count())
        self.assertEqual(reduced_image_collections + 3 * (len(self.dataset_loaders) - 1), Task.objects.all().count())

        # Check explicitly that each of the image collections loaded by the particular loader is gone
        for image_collection_id in self.image_collections[self.dataset_loaders[0]]:
            self.assertEqual(0, ImageCollection.objects.raw({'_id': image_collection_id}).count())

        # Check that each of the image collections loaded by other loaders are still there
        for idx in range(1, len(self.dataset_loaders)):
            for image_collection_id in self.image_collections[self.dataset_loaders[idx]]:
                self.assertEqual(1, ImageCollection.objects.raw({'_id': image_collection_id}).count())

        # Check that each of the tasks associated with the invalidated image collections are removed
        for task_id in self.import_dataset_tasks[self.dataset_loaders[0]]:
            self.assertEqual(0, Task.objects.raw({'_id': task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.dataset_loaders)):
            for task_id in self.import_dataset_tasks[self.dataset_loaders[i]]:
                self.assertEqual(1, Task.objects.raw({'_id': task_id}).count())


class TestInvalidateImageCollection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self):
        # Remove the collections as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()

        # Create systems in two groups
        self.systems = [mock_types.MockSystem() for _ in range(2)]
        for system in self.systems:
            system.save()
        self.unfinished_systems = [mock_types.MockSystem() for _ in range(2)]
        for system in self.unfinished_systems:
            system.save()

        # Make image collections with import tasks
        self.image_collections = []
        self.import_dataset_tasks = {}
        for _ in range(2):
            image_collection = make_image_collection()
            image_collection.save()
            self.image_collections.append(image_collection)

            task = ImportDatasetTask(
                module_name='myloader',
                path='test/filename',
                result=image_collection,
                state=JobState.DONE
            )
            task.save()
            self.import_dataset_tasks[image_collection.identifier] = task.identifier

        # Create run system tasks and trial results on each image collection
        self.run_system_tasks = {}
        self.trial_results = {}
        for image_collection in self.image_collections:
            self.run_system_tasks[image_collection.identifier] = []
            self.trial_results[image_collection.identifier] = []

            for system in self.systems:
                trial_result = mock_types.MockTrialResult(
                        system=system,
                        image_source=image_collection,
                        success=True
                )
                trial_result.save()
                self.trial_results[image_collection.identifier].append(trial_result.identifier)

                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.DONE,
                    result=trial_result
                )
                task.save()
                self.run_system_tasks[image_collection.identifier].append(task.identifier)

            for system in self.unfinished_systems:
                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.UNSTARTED,
                )
                task.save()
                self.run_system_tasks[image_collection.identifier].append(task.identifier)

    def test_invalidate_image_collection_removes_the_collection(self):
        self.assertEqual(len(self.image_collections), ImageCollection.objects.all().count())
        invalidate.invalidate_image_collections([self.image_collections[0].identifier])

        # Check that the number has gone down
        self.assertEqual(len(self.image_collections) - 1, ImageCollection.objects.all().count())

        # Check that the image collection is removed
        self.assertEqual(0, ImageCollection.objects.raw({'_id': self.image_collections[0].identifier}).count())

        # Check that the other collections are still here
        for i in range(1, len(self.image_collections)):
            self.assertEqual(1, ImageCollection.objects.raw({'_id': self.image_collections[i].identifier}).count())

    @unittest.skip("This isn't working, fix it later")
    def test_invalidate_image_collection_removes_images(self):
        # Collect the image ids
        removed_ids = [image._id for image in self.image_collections[0].images]
        kept_ids = [
            image._id
            for image_collection in self.image_collections[1:]
            for image in image_collection.images
        ]

        invalidate.invalidate_image_collections([self.image_collections[0].identifier])

        # Check the expected images are gone
        self.assertEqual(0, Image.objects.raw({'_id': {'$in': removed_ids}}).count())
        self.assertEqual(len(kept_ids), Image.objects.raw({'_id': {'$in': kept_ids}}).count())

    def test_invalidate_image_collection_removes_descendant_trials(self):
        self.assertEqual(len(self.image_collections) * len(self.systems),
                         TrialResult.objects.all().count())
        invalidate.invalidate_image_collections([self.image_collections[0].identifier])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual((len(self.image_collections) - 1) * len(self.systems),
                         TrialResult.objects.all().count())

        # Check explicity that all the descendant trials are removed
        for trial_result_id in self.trial_results[self.image_collections[0].identifier]:
            self.assertEqual(0, TrialResult.objects.raw({'_id': trial_result_id}).count())

        # Check that the other trials are not removed
        for i in range(1, len(self.image_collections)):
            for trial_result_id in self.trial_results[self.image_collections[i].identifier]:
                self.assertEqual(1, TrialResult.objects.raw({'_id': trial_result_id}).count())

    def test_invalidate_image_collection_removes_tasks(self):
        self.assertEqual(len(self.image_collections) * (1 + len(self.systems) + len(self.unfinished_systems)),
                         Task.objects.all().count())
        invalidate.invalidate_image_collections([self.image_collections[0].identifier])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual((len(self.image_collections) - 1) * (1 + len(self.systems) + len(self.unfinished_systems)),
                         Task.objects.all().count())

        # Check explicitly that each of the tasks associated with the invalidated image collection are removed
        self.assertEqual(0, Task.objects.raw({
            '_id': self.import_dataset_tasks[self.image_collections[0].identifier]}).count())
        for run_system_task_id in self.run_system_tasks[self.image_collections[0].identifier]:
            self.assertEqual(0, Task.objects.raw({'_id': run_system_task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.image_collections)):
            self.assertEqual(1, Task.objects.raw({
                '_id': self.import_dataset_tasks[self.image_collections[i].identifier]}).count())
            for run_system_task_id in self.run_system_tasks[self.image_collections[i].identifier]:
                self.assertEqual(1, Task.objects.raw({'_id': run_system_task_id}).count())

    @mock.patch('arvet.batch_analysis.invalidate.autoload_modules')
    def test_invalidate_image_collection_autoloads_image_collection_types(self, mock_autoload):
        invalidate.invalidate_image_collections([self.image_collections[0].pk])
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(ImageSource, [self.image_collections[0].pk]), mock_autoload.call_args_list)


class TestInvalidateSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self):
        # Remove the collections as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

        # Create the basic image sources in two groups
        self.image_collections = [make_image_collection() for _ in range(3)]
        self.unfinisned_image_collections = [make_image_collection() for _ in range(3)]

        # Make systesm
        self.systems = [mock_types.MockSystem() for _ in range(2)]

        # Create run system tasks and trial results
        self.run_system_tasks = {}
        self.trial_results = {}
        for system in self.systems:
            system.save()
            self.run_system_tasks[system.identifier] = []
            self.trial_results[system.identifier] = []

            for image_collection in self.image_collections:
                trial_result = mock_types.MockTrialResult(
                    system=system,
                    image_source=image_collection,
                    success=True
                )
                trial_result.save()
                self.trial_results[system.identifier].append(trial_result.identifier)

                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.DONE,
                    result=trial_result
                )
                task.save()
                self.run_system_tasks[system.identifier].append(task.identifier)

            for image_collection in self.unfinisned_image_collections:
                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.UNSTARTED,
                )
                task.save()
                self.run_system_tasks[system.identifier].append(task.identifier)

    def test_invalidate_system_removes_system(self):
        invalidate.invalidate_systems([self.systems[0].identifier])

        # Check that the system is removed
        self.assertEqual(0, mock_types.MockSystem.objects.raw({'_id': self.systems[0].identifier}).count())

        # Check that the other systems are still here
        for i in range(1, len(self.systems)):
            self.assertEqual(1, mock_types.MockSystem.objects.raw({'_id': self.systems[i].identifier}).count())

    def test_invalidate_system_removes_trials(self):
        invalidate.invalidate_systems([self.systems[0].identifier])

        # Check that all the descendant trials are removed
        for trial_result_id in self.trial_results[self.systems[0].identifier]:
            self.assertEqual(0, TrialResult.objects.raw({'_id': trial_result_id}).count())

        # Check that the other trials are not invalidated
        for i in range(1, len(self.systems)):
            for trial_result_id in self.trial_results[self.systems[i].identifier]:
                self.assertEqual(1, TrialResult.objects.raw({'_id': trial_result_id}).count())

    def test_invalidate_system_removes_tasks(self):
        self.assertEqual((len(self.image_collections) + len(self.unfinisned_image_collections)) * len(self.systems),
                         Task.objects.all().count())
        invalidate.invalidate_systems([self.systems[0].identifier])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual((len(self.image_collections) + len(self.unfinisned_image_collections)) *
                         (len(self.systems) - 1),
                         Task.objects.all().count())

        # Check explicitly that each of the tasks associated with the invalidated system are removed
        for run_system_task_id in self.run_system_tasks[self.systems[0].identifier]:
            self.assertEqual(0, Task.objects.raw({'_id': run_system_task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.systems)):
            for run_system_task_id in self.run_system_tasks[self.systems[i].identifier]:
                self.assertEqual(1, Task.objects.raw({'_id': run_system_task_id}).count())

    @mock.patch('arvet.batch_analysis.invalidate.autoload_modules')
    def test_invalidate_system_autoloads_system_types(self, mock_autoload):
        invalidate.invalidate_systems([self.systems[0].pk])
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(VisionSystem, [self.systems[0].pk]), mock_autoload.call_args_list)

    def test_invalidate_system_by_name_removes_systems(self):
        invalidate.invalidate_systems_by_name(['arvet.core.tests.mock_types.MockSystem'])

        # Check that all the systems are removed
        for system in self.systems:
            self.assertEqual(0, mock_types.MockSystem.objects.raw({'_id': system.identifier}).count())

    def test_invalidate_system_by_name_removes_trials(self):
        invalidate.invalidate_systems_by_name(['arvet.core.tests.mock_types.MockSystem'])

        # Check that all the trials are invalidated
        for system in self.systems:
            for trial_result_id in self.trial_results[system.identifier]:
                self.assertEqual(0, TrialResult.objects.raw({'_id': trial_result_id}).count())

    def test_invalidate_system_by_name_removes_tasks(self):
        invalidate.invalidate_systems_by_name(['arvet.core.tests.mock_types.MockSystem'])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(0, Task.objects.all().count())

        # Check explicitly that each of the tasks associated with the invalidated system are removed
        for system in self.systems:
            for run_system_task_id in self.run_system_tasks[system.identifier]:
                self.assertEqual(0, Task.objects.raw({'_id': run_system_task_id}).count())

    @mock.patch('arvet.batch_analysis.invalidate.autoload_modules')
    def test_invalidate_system_by_name_autoloads_system_types(self, mock_autoload):
        invalidate.invalidate_systems_by_name(['arvet.core.tests.mock_types.MockSystem'])
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(VisionSystem, classes=['arvet.core.tests.mock_types.MockSystem']),
                      mock_autoload.call_args_list)


class TestInvalidateTrial(unittest.TestCase):
    systems = []
    image_collections = []
    metrics = []
    unfinished_metrics = []

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        # Create the basic image sources, systems, and metrics.
        cls.image_collections = [make_image_collection() for _ in range(2)]
        cls.systems = [mock_types.MockSystem() for _ in range(2)]
        cls.metrics = [mock_types.MockMetric() for _ in range(2)]
        cls.unfinished_metrics = [mock_types.MockMetric() for _ in range(2)]

        for system in cls.systems:
            system.save()
        for metric in cls.metrics:
            metric.save()
        for metric in cls.unfinished_metrics:
            metric.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self):
        # Remove the collections as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()

        # Create run system tasks and trial results
        self.run_system_tasks = {}
        self.trial_result_groups = []
        for image_collection in self.image_collections:
            for system in self.systems:
                trial_result_group = []
                for repeat in range(3):
                    trial_result = mock_types.MockTrialResult(
                        system=system,
                        image_source=image_collection,
                        success=True
                    )
                    trial_result.save()
                    trial_result_group.append(trial_result)

                    task = RunSystemTask(
                        system=system,
                        image_source=image_collection,
                        state=JobState.DONE,
                        result=trial_result
                    )
                    task.save()
                    self.run_system_tasks[trial_result.identifier] = task.identifier
                self.trial_result_groups.append(trial_result_group)

        self.measure_trial_tasks = {}
        self.metric_results = {}
        for group_id, trial_result_group in enumerate(self.trial_result_groups):
            self.measure_trial_tasks[group_id] = []
            self.metric_results[group_id] = []
            for metric in self.metrics:
                metric_result = mock_types.MockMetricResult(
                    metric=metric,
                    trial_results=trial_result_group,
                    success=True
                )
                metric_result.save()
                self.metric_results[group_id].append(metric_result.identifier)

                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=trial_result_group,
                    state=JobState.DONE,
                    result=metric_result
                )
                task.save()
                self.measure_trial_tasks[group_id].append(task.identifier)

            for metric in self.unfinished_metrics:
                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=trial_result_group,
                    state=JobState.UNSTARTED
                )
                task.save()
                self.measure_trial_tasks[group_id].append(task.identifier)

    def test_invalidate_trial_removes_trial(self):
        invalidate.invalidate_trial_results([self.trial_result_groups[0][0].identifier])

        # Check that the trial result is removed
        self.assertEqual(0, TrialResult.objects.raw({'_id': self.trial_result_groups[0][0].identifier}).count())

        # Check that the other trial results are still here
        for i in range(len(self.trial_result_groups)):
            for j in range(len(self.trial_result_groups[i])):
                if i is not 0 and j is not 0:
                    self.assertEqual(1, TrialResult.objects.raw({
                        '_id': self.trial_result_groups[i][j].identifier}).count())

    def test_invalidate_trial_removes_metric_results(self):
        invalidate.invalidate_trial_results([self.trial_result_groups[0][0].identifier])

        # Check that all the descendant results are invalidated
        for result_id in self.metric_results[0]:
            self.assertEqual(0, MetricResult.objects.raw({'_id': result_id}).count())

        # Check that the other results are not invalidated
        for i in range(1, len(self.trial_result_groups)):
            for result_id in self.metric_results[i]:
                self.assertEqual(1, MetricResult.objects.raw({'_id': result_id}).count())

    def test_invalidate_trial_removes_tasks(self):
        self.assertEqual(
            sum(len(trial_result_group) for trial_result_group in self.trial_result_groups) +
            len(self.trial_result_groups) * (len(self.metrics) + len(self.unfinished_metrics)),
            Task.objects.all().count()
        )

        # Invalidate the first trial of the first group
        invalidate.invalidate_trial_results([self.trial_result_groups[0][0].identifier])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(
            sum(len(trial_result_group) for trial_result_group in self.trial_result_groups) - 1 +
            (len(self.trial_result_groups) - 1) * (len(self.metrics) + len(self.unfinished_metrics)),
            Task.objects.all().count()
        )

        # Check explicitly that each of the tasks associated with the invalidated trial are removed
        self.assertEqual(0, Task.objects.raw({
            '_id': self.run_system_tasks[self.trial_result_groups[0][0].identifier]
        }).count())
        for task_id in self.measure_trial_tasks[0]:
            self.assertEqual(0, Task.objects.raw({'_id': task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.trial_result_groups[0])):
            self.assertEqual(1, Task.objects.raw({
                '_id': self.run_system_tasks[self.trial_result_groups[0][i].identifier]
            }).count())
        for i in range(1, len(self.trial_result_groups)):
            for trial_result in self.trial_result_groups[i]:
                self.assertEqual(1, Task.objects.raw({
                    '_id': self.run_system_tasks[trial_result.identifier]
                }).count())

            for task_id in self.measure_trial_tasks[i]:
                self.assertEqual(1, Task.objects.raw({'_id': task_id}).count())

    @mock.patch('arvet.batch_analysis.invalidate.autoload_modules')
    def test_invalidate_trial_autoloads_trial_types(self, mock_autoload):
        invalidate.invalidate_trial_results([self.trial_result_groups[0][0].pk])
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(TrialResult, [self.trial_result_groups[0][0].pk]), mock_autoload.call_args_list)


class TestInvalidateMetric(unittest.TestCase):
    systems = []
    image_collections = []
    trials = []
    unfinished_trials = []

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        # Create the basic image sources, systems, and metrics.
        cls.image_collections = [make_image_collection() for _ in range(2)]
        cls.systems = [mock_types.MockSystem() for _ in range(2)]
        for system in cls.systems:
            system.save()

        for image_collection in cls.image_collections:
            for system in cls.systems:
                trial_result = mock_types.MockTrialResult(
                    system=system,
                    image_source=image_collection,
                    success=True
                )
                trial_result.save()
                cls.trials.append(trial_result)

                trial_result = mock_types.MockTrialResult(
                    system=system,
                    image_source=image_collection,
                    success=True
                )
                trial_result.save()
                cls.unfinished_trials.append(trial_result)

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self):
        # Remove the collections as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()

        self.metrics = [mock_types.MockMetric() for _ in range(2)]
        for metric in self.metrics:
            metric.save()

        self.measure_trial_tasks = {}
        self.metric_results = {}
        for metric in self.metrics:
            self.measure_trial_tasks[metric.identifier] = []
            self.metric_results[metric.identifier] = []

            for trial_result in self.trials:
                metric_result = mock_types.MockMetricResult(
                    metric=metric,
                    trial_results=[trial_result],
                    success=True
                )
                metric_result.save()
                self.metric_results[metric.identifier].append(metric_result.identifier)

                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=[trial_result],
                    state=JobState.DONE,
                    result=metric_result
                )
                task.save()
                self.measure_trial_tasks[metric.identifier].append(task.identifier)

            for trial_result in self.unfinished_trials:
                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=[trial_result],
                    state=JobState.UNSTARTED
                )
                task.save()
                self.measure_trial_tasks[metric.identifier].append(task.identifier)

    def test_invalidate_metrics_removes_metric(self):
        invalidate.invalidate_metrics([self.metrics[0].identifier])

        # Check that the metric is removed
        self.assertEqual(0, mock_types.MockMetric.objects.raw({'_id': self.metrics[0].identifier}).count())

        # Check that the other metrics are still here
        for i in range(1, len(self.metrics)):
            self.assertEqual(1, mock_types.MockMetric.objects.raw({'_id': self.metrics[i].identifier}).count())

    def test_invalidate_metrics_removes_results(self):
        invalidate.invalidate_metrics([self.metrics[0].identifier])

        # Check that all the descendant results are invalidated
        for result_id in self.metric_results[self.metrics[0].identifier]:
            self.assertEqual(0, MetricResult.objects.raw({'_id': result_id}).count())

        # Check that the other results are not invalidated
        for i in range(1, len(self.metrics)):
            for result_id in self.metric_results[self.metrics[i].identifier]:
                self.assertEqual(1, MetricResult.objects.raw({'_id': result_id}).count())

    def test_invalidate_benchmark_removes_tasks(self):
        self.assertEqual(
            (len(self.trials) + len(self.unfinished_trials)) * (len(self.metrics)),
            Task.objects.all().count()
        )
        invalidate.invalidate_metrics([self.metrics[0].identifier])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(
            (len(self.trials) + len(self.unfinished_trials)) * (len(self.metrics) - 1),
            Task.objects.all().count()
        )

        # Check explicitly that each of the tasks associated with the invalidated benchmark are removed
        for task_id in self.measure_trial_tasks[self.metrics[0].identifier]:
            self.assertEqual(0, Task.objects.raw({'_id': task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.metrics)):
            for task_id in self.measure_trial_tasks[self.metrics[i].identifier]:
                self.assertEqual(1, Task.objects.raw({'_id': task_id}).count())

    @mock.patch('arvet.batch_analysis.invalidate.autoload_modules')
    def test_invalidate_metric_autoloads_metric_types(self, mock_autoload):
        invalidate.invalidate_metrics([self.metrics[0].pk])
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(Metric, [self.metrics[0].pk]), mock_autoload.call_args_list)


class TestInvalidateMetricResult(unittest.TestCase):
    systems = []
    image_collections = []
    metrics = []
    trials = []

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        # Create the basic image sources, systems, and metrics.
        cls.image_collections = [make_image_collection() for _ in range(2)]
        cls.systems = [mock_types.MockSystem() for _ in range(2)]
        for system in cls.systems:
            system.save()

        cls.metrics = [mock_types.MockMetric() for _ in range(2)]
        for metric in cls.metrics:
            metric.save()

        for image_collection in cls.image_collections:
            for system in cls.systems:
                trial_result = mock_types.MockTrialResult(
                    system=system,
                    image_source=image_collection,
                    success=True
                )
                trial_result.save()
                cls.trials.append(trial_result)

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self):
        # Remove the collections as the start of the test, so that we're sure it's empty
        Task._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()

        self.metric_results = []
        self.measure_trial_tasks = {}
        for metric in self.metrics:
            for trial_result in self.trials:
                metric_result = mock_types.MockMetricResult(
                    metric=metric,
                    trial_results=[trial_result],
                    success=True
                )
                metric_result.save()
                self.metric_results.append(metric_result)

                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=[trial_result],
                    state=JobState.DONE,
                    result=metric_result
                )
                task.save()
                self.measure_trial_tasks[metric_result.identifier] = task.identifier

    def test_invalidate_result_removes_result(self):
        invalidate.invalidate_metric_results([self.metric_results[0].identifier])

        # Check that the metric result is removed
        self.assertEqual(0, MetricResult.objects.raw({'_id': self.metric_results[0].identifier}).count())

        # Check that the other results are still here
        for i in range(1, len(self.metric_results)):
            self.assertEqual(1, MetricResult.objects.raw({'_id': self.metric_results[i].identifier}).count())

    def test_invalidate_result_removes_tasks(self):
        self.assertEqual(len(self.metric_results), Task.objects.all().count())
        invalidate.invalidate_metric_results([self.metric_results[0].identifier])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(len(self.metric_results) - 1, Task.objects.all().count())

        # Check explicitly that each of the tasks associated with the invalidated benchmark are removed
        self.assertEqual(0, Task.objects.raw({
            '_id': self.measure_trial_tasks[self.metric_results[0].identifier]
        }).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.metric_results)):
            self.assertEqual(1, Task.objects.raw({
                '_id': self.measure_trial_tasks[self.metric_results[i].identifier]
            }).count())

    @mock.patch('arvet.batch_analysis.invalidate.autoload_modules')
    def test_invalidate_metric_result_autoloads_result_types(self, mock_autoload):
        invalidate.invalidate_metric_results([self.metric_results[0].pk])
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(MetricResult, [self.metric_results[0].pk]), mock_autoload.call_args_list)


class TestInvalidateIncompleteTasks(unittest.TestCase):
    systems = []
    image_collections = []
    metrics = []

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        # Create the basic image sources, systems, and metrics.
        cls.image_collections = [make_image_collection() for _ in range(2)]
        cls.systems = [mock_types.MockSystem() for _ in range(2)]
        for system in cls.systems:
            system.save()

        cls.metrics = [mock_types.MockMetric() for _ in range(2)]
        for metric in cls.metrics:
            metric.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def setUp(self) -> None:
        # At each test, clear the set of tasks
        Task.objects.all().delete()

    def test_removes_incomplete_import_dataset_tasks(self):
        unstarted_task = ImportDatasetTask(
            module_name='myloader',
            path='test/filename',
            state=JobState.UNSTARTED
        )
        unstarted_task.save()

        running_task = ImportDatasetTask(
            module_name='myloader',
            path='test/filename',
            state=JobState.RUNNING,
            job_id=1,
            node_id='this'
        )
        running_task.save()

        complete = []
        for image_collection in self.image_collections:
            task = ImportDatasetTask(
                module_name='myloader',
                path='test/filename',
                state=JobState.DONE,
                result=image_collection
            )
            task.save()
            complete.append(task)

        self.assertEqual(len(complete) + 2, Task.objects.all().count())
        invalidate.invalidate_incomplete_tasks()
        self.assertEqual(len(complete), Task.objects.all().count())
        self.assertEqual(0, Task.objects.raw({'_id': unstarted_task.pk}).count())
        self.assertEqual(0, Task.objects.raw({'_id': running_task.pk}).count())
        for complete_task in complete:
            self.assertEqual(1, Task.objects.raw({'_id': complete_task.pk}).count())

    def test_removes_incomplete_run_system_tasks(self):
        # Make tasks of each type, some unstarted, some running, some complete
        unstarted = []
        running = []
        complete = []
        for system in self.systems:
            for image_collection in self.image_collections:
                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.UNSTARTED
                )
                task.save()
                unstarted.append(task)

                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.RUNNING,
                    job_id=10,
                    node_id='this'
                )
                task.save()
                running.append(task)

                trial_result = mock_types.MockTrialResult(
                    system=system,
                    image_source=image_collection,
                    success=True
                )
                trial_result.save()
                task = RunSystemTask(
                    system=system,
                    image_source=image_collection,
                    state=JobState.DONE,
                    result=trial_result
                )
                task.save()
                complete.append(task)

        self.assertEqual(len(complete) + len(running) + len(unstarted), Task.objects.all().count())
        invalidate.invalidate_incomplete_tasks()
        self.assertEqual(len(complete), Task.objects.all().count())
        for unstarted_task in unstarted:
            self.assertEqual(0, Task.objects.raw({'_id': unstarted_task.pk}).count())
        for running_task in running:
            self.assertEqual(0, Task.objects.raw({'_id': running_task.pk}).count())
        for complete_task in complete:
            self.assertEqual(1, Task.objects.raw({'_id': complete_task.pk}).count())

        # Clean up after ourselves
        Task.objects.all().delete()
        TrialResult.objects.all().delete()

    def test_removes_incomplete_measure_results(self):
        # Make tasks of each type, some unstarted, some running, some complete
        unstarted = []
        running = []
        complete = []
        for system in self.systems:
            for image_collection in self.image_collections:
                trial_result = mock_types.MockTrialResult(
                    system=system,
                    image_source=image_collection,
                    success=True
                )
                trial_result.save()

                for metric in self.metrics:
                    task = MeasureTrialTask(
                        metric=metric,
                        trial_results=[trial_result],
                        state=JobState.UNSTARTED
                    )
                    task.save()
                    unstarted.append(task)

                    task = MeasureTrialTask(
                        metric=metric,
                        trial_results=[trial_result],
                        state=JobState.RUNNING,
                        job_id=10,
                        node_id='this'
                    )
                    task.save()
                    running.append(task)

                    metric_result = mock_types.MockMetricResult(
                        metric=metric,
                        trial_results=[trial_result],
                        success=True
                    )
                    metric_result.save()
                    task = MeasureTrialTask(
                        metric=metric,
                        trial_results=[trial_result],
                        state=JobState.DONE,
                        result=metric_result
                    )
                    task.save()
                    complete.append(task)

        self.assertEqual(len(complete) + len(running) + len(unstarted), Task.objects.all().count())
        invalidate.invalidate_incomplete_tasks()
        self.assertEqual(len(complete), Task.objects.all().count())
        for unstarted_task in unstarted:
            self.assertEqual(0, Task.objects.raw({'_id': unstarted_task.pk}).count())
        for running_task in running:
            self.assertEqual(0, Task.objects.raw({'_id': running_task.pk}).count())
        for complete_task in complete:
            self.assertEqual(1, Task.objects.raw({'_id': complete_task.pk}).count())

        # Clean up after ourselves
        Task.objects.all().delete()
        MetricResult.objects.all().delete()
        TrialResult.objects.all().delete()


def make_image_collection(length=3) -> ImageCollection:
    """
    A quick helper for making and saving image collections
    :param length:
    :return:
    """
    images = []
    timestamps = []
    for idx in range(length):
        image = mock_types.make_image()
        image.save()
        images.append(image)
        timestamps.append(idx * 0.9)
    image_collection = ImageCollection(
        images=images,
        timestamps=timestamps,
        sequence_type=ImageSequenceType.SEQUENTIAL
    )
    image_collection.save()
    return image_collection
