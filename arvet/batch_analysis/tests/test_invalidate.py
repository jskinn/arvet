import unittest
import unittest.mock as mock
import numpy as np
import bson
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.database.client
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.metadata.image_metadata as imeta
import arvet.core.tests.mock_types as mock_core
import arvet.core.image_collection
import arvet.core.image_entity as ie
import arvet.core.sequence_type
import arvet.core.trial_result
import arvet.core.benchmark
import arvet.database.tests.mock_database_client as mock_client_factory
import arvet.batch_analysis.task
import arvet.batch_analysis.tasks.generate_dataset_task as generate_dataset_task
import arvet.batch_analysis.tasks.import_dataset_task as import_dataset_task
import arvet.batch_analysis.tasks.run_system_task as run_system_task
import arvet.batch_analysis.tasks.benchmark_trial_task as benchmark_trial_task
import arvet.simulation.controllers.trajectory_follow_controller as traj_follow_controller
import arvet.batch_analysis.invalidate as invalidate


class TestInvalidateDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        self.dataset_loaders = ['dataset.loader.loader_A', 'dataset.loader.loader_B']
        self.image_collections = {}
        self.import_dataset_tasks = {}

        for dataset_loader in self.dataset_loaders:
            self.image_collections[dataset_loader] = [make_image_collection(self.mock_db_client).identifier
                                                      for _ in range(2)]

            # Add import dataset tasks for all the image collections
            self.import_dataset_tasks[dataset_loader] = []
            for image_collection_id in self.image_collections[dataset_loader]:
                insert_result = self.mock_db_client.tasks_collection.insert_one(
                    import_dataset_task.ImportDatasetTask(
                        module_name=dataset_loader,
                        path='test/filename',
                        result=image_collection_id,
                        state=arvet.batch_analysis.task.JobState.DONE
                    ).serialize()
                )
                self.import_dataset_tasks[dataset_loader].append(insert_result.inserted_id)

    def test_invalidate_dataset_loader_invalidates_image_collections(self):
        num_image_collections = sum(len(group) for group in self.image_collections.values())

        # Check that we start with the right number of tasks and image collections
        self.assertEqual(num_image_collections, self.mock_db_client.image_source_collection.find().count())
        self.assertEqual(num_image_collections, self.mock_db_client.tasks_collection.find().count())

        with mock.patch('arvet.batch_analysis.invalidate.invalidate_image_collection',
                        wraps=arvet.batch_analysis.invalidate.invalidate_image_collection) as \
                mock_invalidate_image_collection:
            invalidate.invalidate_dataset_loader(self.mock_db_client, self.dataset_loaders[0])

            # Check that all the image collections imported by this loader are invalidated
            for image_collection_id in self.image_collections[self.dataset_loaders[0]]:
                self.assertIn(mock.call(self.mock_db_client, image_collection_id),
                              mock_invalidate_image_collection.call_args_list)

            # Check that the other trials are not invalidated
            for i in range(1, len(self.dataset_loaders)):
                for image_collection_id in self.image_collections[self.dataset_loaders[i]]:
                    self.assertNotIn(mock.call(self.mock_db_client, image_collection_id),
                                     mock_invalidate_image_collection.call_args_list)

        reduced_image_collections = sum(len(self.image_collections[self.dataset_loaders[idx]])
                                        for idx in range(1, len(self.dataset_loaders)))

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(reduced_image_collections, self.mock_db_client.image_source_collection.find().count())
        self.assertEqual(reduced_image_collections, self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the image collections loaded by the particular loader is gone
        for image_collection_id in self.image_collections[self.dataset_loaders[0]]:
            self.assertEqual(0, self.mock_db_client.image_source_collection.find({'_id': image_collection_id}).count())

        # Check that each of the image collections loaded by other loaders are still there
        for idx in range(1, len(self.dataset_loaders)):
            for image_collection_id in self.image_collections[self.dataset_loaders[idx]]:
                self.assertEqual(1, self.mock_db_client.image_source_collection.find(
                    {'_id': image_collection_id}).count())

        # Check that each of the tasks associated with the invalidated image collections are removed
        for task_id in self.import_dataset_tasks[self.dataset_loaders[0]]:
            self.assertEqual(0, self.mock_db_client.tasks_collection.find({'_id': task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.dataset_loaders)):
            for task_id in self.import_dataset_tasks[self.dataset_loaders[i]]:
                self.assertEqual(1, self.mock_db_client.tasks_collection.find({'_id': task_id}).count())


class TestInvalidateImageCollection(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        # Create the basic image sources, systems, and benchmarks.
        self.image_collections = [make_image_collection(self.mock_db_client).identifier for _ in range(2)]
        self.systems = [self.mock_db_client.system_collection.insert_one(mock_core.MockSystem().serialize()).inserted_id
                        for _ in range(2)]
        self.benchmarks = [self.mock_db_client.benchmarks_collection.insert_one(
            mock_core.MockBenchmark().serialize()).inserted_id for _ in range(2)]

        # Add generate dataset tasks for all the image collections.
        self.generate_dataset_tasks = {
            image_collection_id: self.mock_db_client.tasks_collection.insert_one(
                generate_dataset_task.GenerateDatasetTask(
                    bson.ObjectId(), bson.ObjectId(), {}, result=image_collection_id,
                    state=arvet.batch_analysis.task.JobState.DONE).serialize()
            ).inserted_id
            for image_collection_id in self.image_collections
        }

        # Add controllers that follow the image sources
        self.controllers = {
            image_collection_id: self.mock_db_client.image_source_collection.insert_one(
                traj_follow_controller.TrajectoryFollowController(
                    trajectory={}, trajectory_source=image_collection_id,
                    sequence_type=arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL).serialize()
            ).inserted_id
            for image_collection_id in self.image_collections
        }

        # Create run system tasks and trial results
        self.run_system_tasks = {}
        self.trial_results = {}
        for image_collection_id in self.image_collections:
            self.run_system_tasks[image_collection_id] = []
            self.trial_results[image_collection_id] = []

            for system_id in self.systems:
                trial_result_id = self.mock_db_client.trials_collection.insert_one(
                    arvet.core.trial_result.TrialResult(
                        system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
                    ).serialize()
                ).inserted_id
                self.trial_results[image_collection_id].append(trial_result_id)
                self.run_system_tasks[image_collection_id].append(self.mock_db_client.tasks_collection.insert_one(
                    run_system_task.RunSystemTask(system_id, image_collection_id, result=trial_result_id,
                                                  state=arvet.batch_analysis.task.JobState.DONE).serialize()
                ).inserted_id)

    def test_invalidate_image_collection_removes_tasks(self):
        self.assertEqual(len(self.image_collections) * (1 + len(self.systems)),
                         self.mock_db_client.tasks_collection.find().count())
        invalidate.invalidate_image_collection(self.mock_db_client, self.image_collections[0])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual((len(self.image_collections) - 1) * (1 + len(self.systems)),
                         self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the tasks associated with the invalidated image collection are removed
        self.assertEqual(0, self.mock_db_client.tasks_collection.find({
            '_id': self.generate_dataset_tasks[self.image_collections[0]]}).count())
        for run_system_task_id in self.run_system_tasks[self.image_collections[0]]:
            self.assertEqual(0, self.mock_db_client.tasks_collection.find({'_id': run_system_task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.image_collections)):
            self.assertEqual(1, self.mock_db_client.tasks_collection.find({
                '_id': self.generate_dataset_tasks[self.image_collections[i]]}).count())
            for run_system_task_id in self.run_system_tasks[self.image_collections[i]]:
                self.assertEqual(1, self.mock_db_client.tasks_collection.find({'_id': run_system_task_id}).count())

    def test_invalidate_image_collection_invalidates_descendant_trials(self):
        with mock.patch('arvet.batch_analysis.invalidate.invalidate_trial_result') as mock_invalidate_trial:
            invalidate.invalidate_image_collection(self.mock_db_client, self.image_collections[0])

            # Check that all the descendant trials are invalidated
            for trial_result_id in self.trial_results[self.image_collections[0]]:
                self.assertIn(mock.call(self.mock_db_client, trial_result_id),
                              mock_invalidate_trial.call_args_list)

            # Check that the other trials are not invalidated
            for i in range(1, len(self.image_collections)):
                for trial_result_id in self.trial_results[self.image_collections[i]]:
                    self.assertNotIn(mock.call(self.mock_db_client, trial_result_id),
                                     mock_invalidate_trial.call_args_list)

    def test_invalidate_image_collection_invalidates_descendant_controllers(self):
        with mock.patch('arvet.batch_analysis.invalidate.invalidate_controller') as mock_invalidate_controller:
            invalidate.invalidate_image_collection(self.mock_db_client, self.image_collections[0])

            # Check that all the descendant controllers are invalidated
            self.assertIn(mock.call(self.mock_db_client, self.controllers[self.image_collections[0]]),
                          mock_invalidate_controller.call_args_list)

            # Check that the descendant controllers are not invalidated
            for i in range(1, len(self.image_collections)):
                self.assertNotIn(mock.call(self.mock_db_client, self.controllers[self.image_collections[i]]),
                                 mock_invalidate_controller.call_args_list)

    def test_invalidate_image_collection_removes_the_collection(self):
        invalidate.invalidate_image_collection(self.mock_db_client, self.image_collections[0])
        # Check that the image collection is removed
        self.assertEqual(0, self.mock_db_client.image_source_collection.find({
            '_id': self.image_collections[0]}).count())
        # Check that the other collections are still here
        for i in range(1, len(self.image_collections)):
            self.assertEqual(1, self.mock_db_client.image_source_collection.find({
                '_id': self.image_collections[i]}).count())

    def test_invalidate_image_collection_removes_images(self):
        # Collect the image ids
        removed_ids = []
        kept_ids = []
        for s_image_collection in self.mock_db_client.image_source_collection.find({'images': {'$exists': True}},
                                                                                   {'_id': True, 'images': True}):
            if s_image_collection['_id'] == self.image_collections[0]:
                removed_ids += [image_id for _, image_id in s_image_collection['images']]
            else:
                kept_ids += [image_id for _, image_id in s_image_collection['images']]

        invalidate.invalidate_image_collection(self.mock_db_client, self.image_collections[0])

        # Check the removed ids are gone
        for image_id in removed_ids:
            self.assertEqual(0, self.mock_db_client.image_collection.find({'_id': image_id}).count())
        for image_id in kept_ids:
            self.assertEqual(1, self.mock_db_client.image_collection.find({'_id': image_id}).count())


class TestInvalidateController(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        # Add controllers that generate image collections
        self.controllers = [
            self.mock_db_client.image_source_collection.insert_one(
                traj_follow_controller.TrajectoryFollowController(
                    trajectory={}, trajectory_source=None,
                    sequence_type=arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL).serialize()
            ).inserted_id
            for _ in range(2)
        ]

        # Create image sources and generate dataset tasks for the controller
        self.generate_dataset_tasks = {}
        self.image_collections = {}
        for controller_id in self.controllers:
            # Add generate dataset tasks for all the image collections.
            self.image_collections[controller_id] = make_image_collection(self.mock_db_client).identifier
            self.generate_dataset_tasks[controller_id] = self.mock_db_client.tasks_collection.insert_one(
                generate_dataset_task.GenerateDatasetTask(
                    simulator_id=bson.ObjectId(), controller_id=controller_id, simulator_config={},
                    result=self.image_collections[controller_id],
                    state=arvet.batch_analysis.task.JobState.DONE
                ).serialize()
            ).inserted_id

    def test_invalidate_controller_removes_tasks(self):
        self.assertEqual(len(self.image_collections),
                         self.mock_db_client.tasks_collection.find().count())
        invalidate.invalidate_controller(self.mock_db_client, self.controllers[0])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(len(self.image_collections) - 1,
                         self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the tasks associated with the invalidated controller are removed
        self.assertEqual(0, self.mock_db_client.tasks_collection.find({
            '_id': self.generate_dataset_tasks[self.controllers[0]]}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.image_collections)):
            self.assertEqual(1, self.mock_db_client.tasks_collection.find({
                '_id': self.generate_dataset_tasks[self.controllers[i]]}).count())

    def test_invalidate_controller_invalidates_generated_image_collections(self):
        with mock.patch('arvet.batch_analysis.invalidate.invalidate_image_collection') as mock_invalidate_source:
            invalidate.invalidate_controller(self.mock_db_client, self.controllers[0])

            # Check that all the generated image collections are invalidated
            self.assertIn(mock.call(self.mock_db_client, self.image_collections[self.controllers[0]]),
                          mock_invalidate_source.call_args_list)

            # Check that the other image collections are not invalidated
            for i in range(1, len(self.image_collections)):
                self.assertNotIn(mock.call(self.mock_db_client, self.image_collections[self.controllers[i]]),
                                 mock_invalidate_source.call_args_list)

    def test_invalidate_controller_removes_controller(self):
        invalidate.invalidate_controller(self.mock_db_client, self.controllers[0])
        # Check that the controller is removed
        self.assertEqual(0, self.mock_db_client.image_source_collection.find({
            '_id': self.controllers[0]}).count())
        # Check that the other controllers are still here
        for i in range(1, len(self.controllers)):
            self.assertEqual(1, self.mock_db_client.image_source_collection.find({
                '_id': self.controllers[i]}).count())


class TestInvalidateSystem(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        # Create the basic image sources, systems, and benchmarks.
        self.image_collections = [make_image_collection(self.mock_db_client).identifier for _ in range(2)]
        self.systems = [self.mock_db_client.system_collection.insert_one(mock_core.MockSystem().serialize()).inserted_id
                        for _ in range(2)]
        self.benchmarks = [self.mock_db_client.benchmarks_collection.insert_one(
            mock_core.MockBenchmark().serialize()).inserted_id for _ in range(2)]

        # Create run system tasks and trial results
        self.run_system_tasks = {}
        self.trial_results = {}
        for system_id in self.systems:
            self.run_system_tasks[system_id] = []
            self.trial_results[system_id] = []

            for image_collection_id in self.image_collections:
                trial_result_id = self.mock_db_client.trials_collection.insert_one(
                    arvet.core.trial_result.TrialResult(
                        system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
                    ).serialize()
                ).inserted_id
                self.trial_results[system_id].append(trial_result_id)
                self.run_system_tasks[system_id].append(self.mock_db_client.tasks_collection.insert_one(
                    run_system_task.RunSystemTask(system_id, image_collection_id, result=trial_result_id,
                                                  state=arvet.batch_analysis.task.JobState.DONE).serialize()
                ).inserted_id)

    def test_invalidate_system_removes_tasks(self):
        self.assertEqual(len(self.image_collections) * len(self.systems),
                         self.mock_db_client.tasks_collection.find().count())
        invalidate.invalidate_system(self.mock_db_client, self.systems[0])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(len(self.image_collections) * (len(self.systems) - 1),
                         self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the tasks associated with the invalidated system are removed
        for run_system_task_id in self.run_system_tasks[self.systems[0]]:
            self.assertEqual(0, self.mock_db_client.tasks_collection.find({'_id': run_system_task_id}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.systems)):
            for run_system_task_id in self.run_system_tasks[self.systems[i]]:
                self.assertEqual(1, self.mock_db_client.tasks_collection.find({'_id': run_system_task_id}).count())

    def test_invalidate_system_invalidates_trials(self):
        with mock.patch('arvet.batch_analysis.invalidate.invalidate_trial_result') as mock_invalidate_trial:
            invalidate.invalidate_system(self.mock_db_client, self.systems[0])

            # Check that all the descendant trials are invalidated
            for trial_result_id in self.trial_results[self.systems[0]]:
                self.assertIn(mock.call(self.mock_db_client, trial_result_id),
                              mock_invalidate_trial.call_args_list)

            # Check that the other trials are not invalidated
            for i in range(1, len(self.systems)):
                for trial_result_id in self.trial_results[self.systems[i]]:
                    self.assertNotIn(mock.call(self.mock_db_client, trial_result_id),
                                     mock_invalidate_trial.call_args_list)

    def test_invalidate_system_removes_system(self):
        invalidate.invalidate_system(self.mock_db_client, self.systems[0])
        # Check that the system is removed
        self.assertEqual(0, self.mock_db_client.system_collection.find({
            '_id': self.systems[0]}).count())
        # Check that the other systems are still here
        for i in range(1, len(self.systems)):
            self.assertEqual(1, self.mock_db_client.system_collection.find({
                '_id': self.systems[i]}).count())


class TestInvalidateTrial(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        # Create the basic image sources, systems, and benchmarks.
        self.image_collections = [make_image_collection(self.mock_db_client).identifier for _ in range(2)]
        self.systems = [self.mock_db_client.system_collection.insert_one(mock_core.MockSystem().serialize()).inserted_id
                        for _ in range(2)]
        self.benchmarks = [self.mock_db_client.benchmarks_collection.insert_one(
            mock_core.MockBenchmark().serialize()).inserted_id for _ in range(2)]

        # Create run system tasks and trial results
        self.run_system_tasks = {}
        self.trial_results = []
        for image_collection_id in self.image_collections:
            for system_id in self.systems:
                trial_result_id = self.mock_db_client.trials_collection.insert_one(
                    arvet.core.trial_result.TrialResult(
                        system_id, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}
                    ).serialize()
                ).inserted_id
                self.trial_results.append(trial_result_id)
                self.run_system_tasks[trial_result_id] = self.mock_db_client.tasks_collection.insert_one(
                    run_system_task.RunSystemTask(system_id, image_collection_id, result=trial_result_id,
                                                  state=arvet.batch_analysis.task.JobState.DONE).serialize()
                ).inserted_id

        self.benchmark_trial_tasks = {}
        self.benchmark_results = {}
        for trial_result_id in self.trial_results:
            self.benchmark_trial_tasks[trial_result_id] = []
            self.benchmark_results[trial_result_id] = []
            for benchmark_id in self.benchmarks:
                result_id = self.mock_db_client.results_collection.insert_one(
                    arvet.core.benchmark.BenchmarkResult(benchmark_id, trial_result_id, True).serialize()).inserted_id
                self.benchmark_results[trial_result_id].append(result_id)
                self.benchmark_trial_tasks[trial_result_id].append(
                    self.mock_db_client.tasks_collection.insert_one(
                        benchmark_trial_task.BenchmarkTrialTask(
                            trial_result_id, benchmark_id,
                            result=result_id, state=arvet.batch_analysis.task.JobState.DONE).serialize()
                    ).inserted_id
                )

    def test_invalidate_trial_removes_tasks(self):
        self.assertEqual(len(self.trial_results) * (1 + len(self.benchmarks)),
                         self.mock_db_client.tasks_collection.find().count())
        invalidate.invalidate_trial_result(self.mock_db_client, self.trial_results[0])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual((len(self.trial_results) - 1) * (1 + len(self.benchmarks)),
                         self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the tasks associated with the invalidated trial are removed
        self.assertEqual(0, self.mock_db_client.tasks_collection.find({
            '_id': self.run_system_tasks[self.trial_results[0]]}).count())
        for benchmark_task in self.benchmark_trial_tasks[self.trial_results[0]]:
            self.assertEqual(0, self.mock_db_client.tasks_collection.find({'_id': benchmark_task}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.trial_results)):
            self.assertEqual(1, self.mock_db_client.tasks_collection.find({
                '_id': self.run_system_tasks[self.trial_results[i]]}).count())
            for benchmark_task in self.benchmark_trial_tasks[self.trial_results[i]]:
                self.assertEqual(1, self.mock_db_client.tasks_collection.find({'_id': benchmark_task}).count())

    def test_invalidate_trial_invalidates_results(self):
        with mock.patch('arvet.batch_analysis.invalidate.invalidate_benchmark_result') as mock_invalidate_result:
            invalidate.invalidate_trial_result(self.mock_db_client, self.trial_results[0])

            # Check that all the descendant results are invalidated
            for result_id in self.benchmark_results[self.trial_results[0]]:
                self.assertIn(mock.call(self.mock_db_client, result_id),
                              mock_invalidate_result.call_args_list)

            # Check that the other results are not invalidated
            for i in range(1, len(self.trial_results)):
                for result_id in self.benchmark_results[self.trial_results[i]]:
                    self.assertNotIn(mock.call(self.mock_db_client, result_id),
                                     mock_invalidate_result.call_args_list)

    def test_invalidate_trial_removes_trial(self):
        invalidate.invalidate_trial_result(self.mock_db_client, self.trial_results[0])
        # Check that the image collection is removed
        self.assertEqual(0, self.mock_db_client.trials_collection.find({
            '_id': self.trial_results[0]}).count())
        # Check that the other collections are still here
        for i in range(1, len(self.trial_results)):
            self.assertEqual(1, self.mock_db_client.trials_collection.find({
                '_id': self.trial_results[i]}).count())


class TestInvalidateBenchmark(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        # Create the basic trial results and benchmarks
        self.trial_results = [self.mock_db_client.trials_collection.insert_one(
            arvet.core.trial_result.TrialResult(
                bson.ObjectId(), True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {}).serialize()
        ).inserted_id for _ in range(2)]
        self.benchmarks = [self.mock_db_client.benchmarks_collection.insert_one(
            mock_core.MockBenchmark().serialize()).inserted_id for _ in range(2)]

        self.benchmark_trial_tasks = {}
        self.benchmark_results = {}

        # Create the benchmark results and tasks
        for benchmark_id in self.benchmarks:
            self.benchmark_trial_tasks[benchmark_id] = []
            self.benchmark_results[benchmark_id] = []
            for trial_result_id in self.trial_results:
                result_id = self.mock_db_client.results_collection.insert_one(
                    arvet.core.benchmark.BenchmarkResult(benchmark_id, trial_result_id, True).serialize()).inserted_id
                self.benchmark_results[benchmark_id].append(result_id)
                self.benchmark_trial_tasks[benchmark_id].append(
                    self.mock_db_client.tasks_collection.insert_one(
                        benchmark_trial_task.BenchmarkTrialTask(
                            trial_result_id, benchmark_id,
                            result=result_id, state=arvet.batch_analysis.task.JobState.DONE).serialize()
                    ).inserted_id
                )

    def test_invalidate_benchmark_removes_tasks(self):
        self.assertEqual(len(self.trial_results) * (len(self.benchmarks)),
                         self.mock_db_client.tasks_collection.find().count())
        invalidate.invalidate_benchmark(self.mock_db_client, self.benchmarks[0])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(len(self.trial_results) * (len(self.benchmarks) - 1),
                         self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the tasks associated with the invalidated benchmark are removed
        for benchmark_task in self.benchmark_trial_tasks[self.benchmarks[0]]:
            self.assertEqual(0, self.mock_db_client.tasks_collection.find({'_id': benchmark_task}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.benchmarks)):
            for benchmark_task in self.benchmark_trial_tasks[self.benchmarks[i]]:
                self.assertEqual(1, self.mock_db_client.tasks_collection.find({'_id': benchmark_task}).count())

    def test_invalidate_benchmark_invalidates_results(self):
        with mock.patch('arvet.batch_analysis.invalidate.invalidate_benchmark_result') as mock_invalidate_result:
            invalidate.invalidate_benchmark(self.mock_db_client, self.benchmarks[0])

            # Check that all the descendant results are invalidated
            for result_id in self.benchmark_results[self.benchmarks[0]]:
                self.assertIn(mock.call(self.mock_db_client, result_id),
                              mock_invalidate_result.call_args_list)

            # Check that the other results are not invalidated
            for i in range(1, len(self.benchmarks)):
                for result_id in self.benchmark_results[self.benchmarks[i]]:
                    self.assertNotIn(mock.call(self.mock_db_client, result_id),
                                     mock_invalidate_result.call_args_list)

    def test_invalidate_benchmark_removes_benchmark(self):
        invalidate.invalidate_benchmark(self.mock_db_client, self.benchmarks[0])
        # Check that the image collection is removed
        self.assertEqual(0, self.mock_db_client.benchmarks_collection.find({
            '_id': self.benchmarks[0]}).count())
        # Check that the other collections are still here
        for i in range(1, len(self.benchmarks)):
            self.assertEqual(1, self.mock_db_client.benchmarks_collection.find({
                '_id': self.benchmarks[i]}).count())


class TestInvalidateResult(unittest.TestCase):

    def setUp(self):
        self.zombie_db_client = mock_client_factory.create()
        self.mock_db_client = self.zombie_db_client.mock

        # Create benchmark results and tasks
        self.benchmark_trial_tasks = {}
        self.benchmark_results = []
        for _ in range(2):
            result_id = self.mock_db_client.results_collection.insert_one(
                arvet.core.benchmark.BenchmarkResult(bson.ObjectId(), bson.ObjectId(), True).serialize()).inserted_id
            self.benchmark_results.append(result_id)
            self.benchmark_trial_tasks[result_id] = self.mock_db_client.tasks_collection.insert_one(
                benchmark_trial_task.BenchmarkTrialTask(
                    bson.ObjectId(), bson.ObjectId(),
                    result=result_id, state=arvet.batch_analysis.task.JobState.DONE).serialize()
            ).inserted_id

    def test_invalidate_result_removes_tasks(self):
        self.assertEqual(len(self.benchmark_trial_tasks),
                         self.mock_db_client.tasks_collection.find().count())
        invalidate.invalidate_benchmark_result(self.mock_db_client, self.benchmark_results[0])

        # Check that the total number of tasks has gone down like we expected
        self.assertEqual(len(self.benchmark_trial_tasks) - 1,
                         self.mock_db_client.tasks_collection.find().count())

        # Check explicitly that each of the tasks associated with the invalidated benchmark are removed
        self.assertEqual(0, self.mock_db_client.tasks_collection.find({
            '_id': self.benchmark_trial_tasks[self.benchmark_results[0]]}).count())

        # Check that the remaining tasks are still there.
        for i in range(1, len(self.benchmark_results)):
            self.assertEqual(1, self.mock_db_client.tasks_collection.find({
                '_id': self.benchmark_trial_tasks[self.benchmark_results[i]]}).count())

    def test_invalidate_result_removes_result(self):
        invalidate.invalidate_benchmark_result(self.mock_db_client, self.benchmark_results[0])
        # Check that the image collection is removed
        self.assertEqual(0, self.mock_db_client.results_collection.find({
            '_id': self.benchmark_results[0]}).count())
        # Check that the other collections are still here
        for i in range(1, len(self.benchmark_results)):
            self.assertEqual(1, self.mock_db_client.results_collection.find({
                '_id': self.benchmark_results[i]}).count())


def make_image_collection(db_client: arvet.database.client.DatabaseClient, length=3)\
        -> arvet.core.image_collection.ImageCollection:
    images = {}
    for i in range(length):
        image_entity = make_image(i)
        image_entity.save_image_data(db_client)
        images[i] = db_client.image_collection.insert_one(image_entity.serialize()).inserted_id
    image_collection = arvet.core.image_collection.ImageCollection(
        images=images, db_client_=db_client, type_=arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
    image_collection.refresh_id(db_client.image_source_collection.insert_one(image_collection.serialize()).inserted_id)
    return image_collection


def make_image(index=1, **kwargs) -> arvet.core.image_entity.ImageEntity:
    kwargs = du.defaults(kwargs, {
        'id_': bson.objectid.ObjectId(),
        'data': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': imeta.ImageMetadata(
            hash_=b'\xf1\x9a\xe2|' + np.random.randint(0, 0xFFFFFFFF).to_bytes(4, 'big'),
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=tf.Transform(location=(1 + 100 * index, 2 + np.random.uniform(-1, 1), 3),
                                     rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
            intrinsics=cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
            environment_type=imeta.EnvironmentType.INDOOR_CLOSE,
            light_level=imeta.LightingLevel.WELL_LIT, time_of_day=imeta.TimeOfDay.DAY,
            lens_focal_distance=5, aperture=22, simulation_world='TestSimulationWorld',
            lighting_model=imeta.LightingModel.LIT, texture_mipmap_bias=1,
            normal_maps_enabled=2, roughness_enabled=True, geometry_decimation=0.8,
            procedural_generation_seed=16234, labelled_objects=(
                imeta.LabelledObject(class_names=('car',), bounding_box=(12, 144, 67, 43),
                                     label_color=(123, 127, 112),
                                     relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)),
                                     object_id='Car-002'),
                imeta.LabelledObject(class_names=('cat',), bounding_box=(125, 244, 117, 67),
                                     label_color=(27, 89, 62),
                                     relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                                     object_id='cat-090')
            ), average_scene_depth=90.12),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth_data': np.random.uniform(0, 1, (32, 32)),
        'labels_data': np.random.uniform(0, 1, (32, 32, 3)),
        'world_normals_data': np.random.uniform(0, 1, (32, 32, 3))
    })
    return ie.ImageEntity(**kwargs)
