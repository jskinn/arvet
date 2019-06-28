# Copyright (c) 2017, John Skinner
import os
import unittest
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image
from arvet.core.image_collection import ImageCollection
from arvet.core.metric import Metric

from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask

import arvet.core.tests.mock_types as mock_types
import arvet.batch_analysis.experiment as ex


# A minimal experiment, so we can instantiate the abstract class
class MockExperiment(ex.Experiment):

    def __init__(self, name="MockExperiment", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def do_imports(self, path_manager):
        pass

    def schedule_tasks(self):
        pass


class TestExperimentDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        ex.Experiment._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        ex.Experiment._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = MockExperiment()
        obj.save()

        # Load all the entities
        all_entities = list(MockExperiment.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()


class TestRunAllDatabase(unittest.TestCase):
    systems = None
    image_sources = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Ensure we have a clean slate in the database
        Task._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

        cls.systems = [mock_types.MockSystem() for _ in range(2)]
        for system in cls.systems:
            system.save()
        cls.image_sources = [make_image_collection() for _ in range(2)]

    def tearDown(self) -> None:
        # Ensure there are no tasks left at the end of each test
        Task._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collections for all the models used
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_makes_run_system_tasks(self):
        repeats = 2
        self.assertEqual(0, RunSystemTask.objects.all().count())
        _, pending = ex.run_all(self.systems, self.image_sources, repeats)

        self.assertEqual(len(self.systems) * len(self.image_sources) * repeats, pending)
        for system in self.systems:
            for image_source in self.image_sources:
                for repeat in range(repeats):
                    self.assertEqual(1, RunSystemTask.objects.raw({
                        'system': system.identifier,
                        'image_source': image_source.identifier,
                        'repeat': repeat
                    }).count())

    def test_passes_through_task_settings(self):
        num_cpus = 3
        num_gpus = 2
        memory_requirements = '64K'
        expected_duration = '6:45:12'
        repeats = 2
        self.assertEqual(0, RunSystemTask.objects.all().count())
        _, pending = ex.run_all(
            self.systems, self.image_sources, repeats,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_requirements=memory_requirements,
            expected_duration=expected_duration
        )

        self.assertEqual(len(self.systems) * len(self.image_sources) * repeats, pending)
        for system in self.systems:
            for image_source in self.image_sources:
                for repeat in range(repeats):
                    task = RunSystemTask.objects.get({
                        'system': system.identifier,
                        'image_source': image_source.identifier,
                        'repeat': repeat
                    })
                    self.assertEqual(num_cpus, task.num_cpus)
                    self.assertEqual(num_gpus, task.num_gpus)
                    self.assertEqual(memory_requirements, task.memory_requirements)
                    self.assertEqual(expected_duration, task.expected_duration)

    def test_run_all_filters_by_allowed_image_sources(self):
        # Make it so that some systems cannot run with some image sources
        self.systems[0].sources_blacklist.append(self.image_sources[0].identifier)
        self.systems[1].sources_blacklist.append(self.image_sources[1].identifier)
        repeats = 2
        self.assertEqual(0, RunSystemTask.objects.all().count())
        _, pending = ex.run_all(self.systems, self.image_sources, repeats)

        # Check that only those combinations where the image source is appropriate for the system are scheduled.
        self.assertEqual((len(self.systems) * len(self.image_sources) - 2) * repeats, pending)
        for sys_idx in range(len(self.systems)):
            for image_source_idx in range(len(self.image_sources)):
                for repeat in range(repeats):
                    if sys_idx == image_source_idx == 0 or sys_idx == image_source_idx == 1:
                        self.assertEqual(0, RunSystemTask.objects.raw({
                            'system': self.systems[sys_idx].identifier,
                            'image_source': self.image_sources[image_source_idx].identifier,
                            'repeat': repeat
                        }).count())
                    else:
                        self.assertEqual(1, RunSystemTask.objects.raw({
                            'system': self.systems[sys_idx].identifier,
                            'image_source': self.image_sources[image_source_idx].identifier,
                            'repeat': repeat
                        }).count())

    def test_doesnt_return_incomplete_results(self):
        repeats = 2

        # Create some existing tasks that are incomplete
        for system in self.systems:
            for image_source in self.image_sources:
                for repeat in range(repeats):
                    task = RunSystemTask(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        state=JobState.RUNNING
                    )
                    task.save()

        existing_results, pending = ex.run_all(self.systems, self.image_sources, repeats)
        self.assertEqual(len(self.systems) * len(self.image_sources) * repeats, pending)
        self.assertEqual({}, existing_results)

    def test_returns_completed_results(self):
        repeats = 3

        # Create some existing trial results with complete run system tasks
        num_complete_trials = 0
        trial_results = {}
        for system in self.systems[1:]:
            trial_results[system.identifier] = {}
            for image_source in self.image_sources[1:]:
                trial_results[system.identifier][image_source.identifier] = []
                for repeat in range(repeats - 1):
                    trial_result = mock_types.MockTrialResult(
                        system=system,
                        image_source=image_source,
                        success=True
                    )
                    trial_result.save()
                    trial_results[system.identifier][image_source.identifier].append(trial_result.identifier)
                    num_complete_trials += 1

                    task = RunSystemTask(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        state=JobState.DONE,
                        result=trial_result
                    )
                    task.save()

        existing_results, pending = ex.run_all(self.systems, self.image_sources, repeats)
        self.assertEqual(len(self.systems) * len(self.image_sources) * repeats - num_complete_trials, pending)

        for system in self.systems[1:]:
            for image_source in self.image_sources[1:]:
                for repeat in range(repeats - 1):
                    self.assertEqual(
                        trial_results[system.identifier][image_source.identifier][repeat],
                        existing_results[system.identifier][image_source.identifier][repeat].identifier
                    )


@unittest.skip("Not running profiling")
class TestRunAllDatabaseProfile(unittest.TestCase):
    systems = None
    image_sources = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Ensure we have a clean slate in the database
        Task._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collections for all the models used
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_profile(self):
        import cProfile as profile

        # Create systems and image sources
        systems = [mock_types.MockSystem() for _ in range(5)]
        for system in systems:
            system.save()
        image_sources = [make_image_collection() for _ in range(5)]
        repeats = 10
        self.assertEqual(0, RunSystemTask.objects.all().count())

        stats_file = "run_all.prof"
        profile.runctx("ex.run_all(systems, image_sources, repeats)",
                       locals=locals(), globals=globals(), filename=stats_file)


class TestMeasureAllDatabase(unittest.TestCase):
    systems = None
    image_sources = None
    metrics = None
    trial_results = None
    trial_result_groups = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Ensure we have a clean slate in the database
        Task._mongometa.collection.drop()
        mock_types.MockMetricResult._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        Metric._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

        cls.systems = [mock_types.MockSystem() for _ in range(2)]
        for system in cls.systems:
            system.save()
        cls.metrics = [mock_types.MockMetric() for _ in range(2)]
        for metric in cls.metrics:
            metric.save()
        cls.image_sources = [make_image_collection() for _ in range(2)]

        # Make trial results for systems and image sources
        cls.trial_results = {}
        for system in cls.systems:
            cls.trial_results[system.identifier] = {}
            for image_source in cls.image_sources:
                cls.trial_results[system.identifier][image_source.identifier] = []
                for repeat in range(2):
                    trial_result = mock_types.MockTrialResult(
                        system=system,
                        image_source=image_source,
                        success=True
                    )
                    trial_result.save()
                    cls.trial_results[system.identifier][image_source.identifier].append(trial_result)

                    task = RunSystemTask(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        state=JobState.DONE,
                        result=trial_result
                    )
                    task.save()

        cls.trial_result_groups = [trial_result_group
                               for inner_group in cls.trial_results.values()
                               for trial_result_group in inner_group.values()]

    def tearDown(self) -> None:
        # Ensure there are no tasks left at the end of each test
        Task._mongometa.collection.drop()
        mock_types.MockMetricResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collections for all the models used
        mock_types.MockTrialResult._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        Metric._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_makes_measure_trial_tasks(self):
        self.assertEqual(0, MeasureTrialTask.objects.all().count())

        _, pending = ex.measure_all(self.metrics, self.trial_result_groups)

        self.assertEqual(len(self.metrics) * len(self.trial_result_groups), pending)
        for metric in self.metrics:
            for trial_result_group in self.trial_result_groups:
                self.assertEqual(1, MeasureTrialTask.objects.raw({
                    'metric': metric.identifier,
                    'trial_results': {'$all': [tr.identifier for tr in trial_result_group]}
                }).count())

    def test_passes_through_task_settings(self):
        num_cpus = 3
        num_gpus = 2
        memory_requirements = '64K'
        expected_duration = '6:45:12'
        self.assertEqual(0, MeasureTrialTask.objects.all().count())
        _, pending = ex.measure_all(
            self.metrics, self.trial_result_groups,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_requirements=memory_requirements,
            expected_duration=expected_duration
        )

        self.assertEqual(len(self.metrics) * len(self.trial_result_groups), pending)
        for metric in self.metrics:
            for trial_result_group in self.trial_result_groups:
                task = MeasureTrialTask.objects.get({
                    'metric': metric.identifier,
                    'trial_results': {'$all': [tr.identifier for tr in trial_result_group]}
                })
                self.assertEqual(num_cpus, task.num_cpus)
                self.assertEqual(num_gpus, task.num_gpus)
                self.assertEqual(memory_requirements, task.memory_requirements)
                self.assertEqual(expected_duration, task.expected_duration)

    def test_run_all_filters_by_allowed_image_sources(self):
        # Make it so that some systems cannot run with some image sources
        self.metrics[0].trials_blacklist.extend(tr.identifier for tr in self.trial_result_groups[0])
        self.metrics[1].trials_blacklist.extend(tr.identifier for tr in self.trial_result_groups[1])

        _, pending = ex.measure_all(self.metrics, self.trial_result_groups)

        # Check that only those combinations where the image source is appropriate for the system are scheduled.
        self.assertEqual(len(self.metrics) * len(self.trial_result_groups) - 2, pending)
        for mtr_idx in range(len(self.metrics)):
            for tr_idx in range(len(self.trial_result_groups)):
                if mtr_idx == tr_idx == 0 or mtr_idx == tr_idx == 1:
                    self.assertEqual(0, MeasureTrialTask.objects.raw({
                        'metric': self.metrics[mtr_idx].identifier,
                        'trial_results': {'$all': [tr.identifier for tr in self.trial_result_groups[tr_idx]]}
                    }).count())
                else:
                    self.assertEqual(1, MeasureTrialTask.objects.raw({
                        'metric': self.metrics[mtr_idx].identifier,
                        'trial_results': {'$all': [tr.identifier for tr in self.trial_result_groups[tr_idx]]}
                    }).count())

    def test_doesnt_return_incomplete_results(self):
        # Create some existing measure trials tasks that are incomplete
        for metric in self.metrics:
            for tr_group in self.trial_result_groups:
                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=tr_group,
                    state=JobState.RUNNING
                )
                task.save()

        existing_results, pending = ex.measure_all(self.metrics, self.trial_result_groups)
        self.assertEqual(len(self.metrics) * len(self.trial_result_groups), pending)
        self.assertEqual({}, existing_results)

    def test_returns_completed_results(self):
        # Create some existing trial results with complete run system tasks
        num_complete_measures = 0
        metric_results = {}
        for metric in self.metrics[1:]:
            metric_results[metric.identifier] = {}
            for tr_idx, tr_group in enumerate(self.trial_result_groups[1:]):
                metric_result = mock_types.MockMetricResult(
                    metric=metric,
                    trial_results=tr_group,
                    success=True
                )
                metric_result.save()
                metric_results[metric.identifier][tr_idx] = metric_result.identifier
                num_complete_measures += 1

                task = MeasureTrialTask(
                    metric=metric,
                    trial_results=tr_group,
                    state=JobState.DONE,
                    result=metric_result
                )
                task.save()

        existing_results, pending = ex.measure_all(self.metrics, self.trial_result_groups)
        self.assertEqual(len(self.metrics) * len(self.trial_result_groups) - num_complete_measures, pending)

        for metric in self.metrics[1:]:
            for tr_idx in range(len(self.trial_result_groups) - 1):
                self.assertEqual(
                    metric_results[metric.identifier][tr_idx],
                    existing_results[metric.identifier][tr_idx].identifier
                )


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
