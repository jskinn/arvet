# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
from pymodm.errors import ValidationError
from pymodm.context_managers import no_auto_dereference
from bson import ObjectId
import arvet.database.tests.database_connection as dbconn
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.metric import Metric
import arvet.core.tests.mock_types as mock_core
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask
from arvet.batch_analysis.simple_experiment import SimpleExperiment


class CountedSystem(mock_core.MockSystem):
    instances = 0

    def __init__(self, *args, **kwargs):
        super(CountedSystem, self).__init__(*args, **kwargs)
        CountedSystem.instances += 1


class CountedImageSource(mock_core.MockImageSource):
    instances = 0

    def __init__(self, *args, **kwargs):
        super(CountedImageSource, self).__init__(*args, **kwargs)
        CountedImageSource.instances += 1


class CountedMetric(mock_core.MockMetric):
    instances = 0

    def __init__(self, *args, **kwargs):
        super(CountedMetric, self).__init__(*args, **kwargs)
        CountedMetric.instances += 1


class TestExperimentDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        SimpleExperiment._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        SimpleExperiment._mongometa.collection.drop()
        mock_core.MockMetric._mongometa.collection.drop()
        mock_core.MockSystem._mongometa.collection.drop()
        mock_core.MockImageSource._mongometa.collection.drop()

    def test_stores_and_loads(self):
        system = mock_core.MockSystem()
        system.save()

        image_source = mock_core.MockImageSource()
        image_source.save()

        metric = mock_core.MockMetric()
        metric.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            enabled=True,
            systems=[system],
            image_sources=[image_source],
            metrics=[metric]
        )
        obj.save()

        # Load all the entities
        all_entities = list(SimpleExperiment.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)

        # Clean up
        SimpleExperiment.objects.all().delete()

    def test_stores_and_loads_minimal(self):
        obj = SimpleExperiment(name="TestSimpleExperiment")
        obj.save()

        # Load all the entities
        all_entities = list(SimpleExperiment.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)

        # Clean up
        SimpleExperiment.objects.all().delete()

    def test_required_fields_are_required(self):
        # No name
        obj = SimpleExperiment()
        with self.assertRaises(ValidationError):
            obj.save()

    @mock.patch('arvet.batch_analysis.simple_experiment.autoload_modules', autospec=True)
    def test_load_referenced_models_autoloads_models_that_are_just_ids(self, mock_autoload):
        system = mock_core.MockSystem()
        system.save()

        image_source = mock_core.MockImageSource()
        image_source.save()

        metric = mock_core.MockMetric()
        metric.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            enabled=True,
            systems=[system],
            image_sources=[image_source],
            metrics=[metric]
        )
        obj.save()
        del obj     # Clear existing references, which should reset the references to ids

        obj = next(SimpleExperiment.objects.all())
        self.assertFalse(mock_autoload.called)
        obj.load_referenced_models()
        self.assertTrue(mock_autoload.called)
        self.assertIn(mock.call(VisionSystem, ids=[system.identifier]), mock_autoload.call_args_list)
        self.assertIn(mock.call(ImageSource, ids=[image_source.identifier]), mock_autoload.call_args_list)
        self.assertIn(mock.call(Metric, ids=[metric.identifier]), mock_autoload.call_args_list)

    @mock.patch('arvet.batch_analysis.simple_experiment.autoload_modules', autospec=True)
    def test_load_referenced_models_does_nothing_to_models_that_are_already_objects(self, mock_autoload):
        system = mock_core.MockSystem()
        system.save()

        image_source = mock_core.MockImageSource()
        image_source.save()

        metric = mock_core.MockMetric()
        metric.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            enabled=True,
            systems=[system],
            image_sources=[image_source],
            metrics=[metric]
        )
        obj.save()

        self.assertFalse(mock_autoload.called)
        obj.load_referenced_models()
        self.assertFalse(mock_autoload.called)

    def test_add_vision_systems_enforces_uniqueness_without_dereferencing(self):
        system1 = CountedSystem()
        system1.save()

        system2 = CountedSystem()
        system2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            enabled=True,
            systems=[system1]
        )
        obj.save()

        del obj  # Clear the obj to clear existing objects and references
        CountedSystem.instances = 0

        obj = next(SimpleExperiment.objects.all())
        self.assertEqual(0, CountedSystem.instances)
        obj.add_vision_systems([system1, system2])
        self.assertEqual(0, CountedSystem.instances)
        # this will auto-dereference
        self.assertEqual(obj.systems, [system1, system2])

        # check we can still save
        obj.save()

        # Clean up
        SimpleExperiment.objects.all().delete()

    def test_add_image_sources_enforces_uniqueness_without_dereferencing(self):
        image_source1 = CountedImageSource()
        image_source1.save()

        image_source2 = CountedImageSource()
        image_source2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            enabled=True,
            image_sources=[image_source1]
        )
        obj.save()

        del obj  # Clear the obj to clear existing objects and references
        CountedImageSource.instances = 0

        obj = next(SimpleExperiment.objects.all())
        self.assertEqual(0, CountedImageSource.instances)
        obj.add_image_sources([image_source1, image_source2])
        self.assertEqual(0, CountedImageSource.instances)
        # this will auto-dereference
        self.assertEqual(obj.image_sources, [image_source1, image_source2])

        # check we can still save
        obj.save()

        # Clean up
        SimpleExperiment.objects.all().delete()

    def test_add_metrics_enforces_uniqueness_without_dereferencing(self):
        metric1 = CountedMetric()
        metric1.save()

        metric2 = CountedMetric()
        metric2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            enabled=True,
            metrics=[metric1]
        )
        obj.save()

        del obj  # Clear the obj to clear existing objects and references
        CountedMetric.instances = 0

        obj = next(SimpleExperiment.objects.all())
        self.assertEqual(0, CountedMetric.instances)
        obj.add_metrics([metric1, metric2])
        self.assertEqual(0, CountedMetric.instances)
        # this will auto-dereference
        self.assertEqual(obj.metrics, [metric1, metric2])

        # check we can still save
        obj.save()

        # Clean up
        SimpleExperiment.objects.all().delete()


class TestSimpleExperiment(unittest.TestCase):

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_runs_all_systems_with_all_image_sources_repeatedly(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metric = mock_core.MockMetric(_id=ObjectId())
        repeats = 2

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=[metric],
            repeats=repeats
        )
        subject.schedule_tasks()

        for system in systems:
            for image_source in image_sources:
                for repeat in range(repeats):
                    self.assertIn(mock.call(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        num_cpus=mock.ANY, num_gpus=mock.ANY,
                        memory_requirements=mock.ANY,
                        expected_duration=mock.ANY
                    ), mock_task_manager.get_run_system_task.call_args_list)

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_uses_run_system_tasks_correctly(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(1)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(1)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(1)]
        repeats = 1

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        # Note: This is a separate test because autospecs are expensive in a loop
        make_mock_get_run_system_task(mock_task_manager, trial_results_map, autospec=True)

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats
        )
        subject.schedule_tasks()

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_measures_all_trial_results(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(3)]
        repeats = 3

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        trial_groups = [
            trial_group
            for trials_by_source in trial_results_map.values()
            for trial_group in trials_by_source.values()
        ]
        make_mock_get_run_system_task(mock_task_manager, trial_results_map, autospec=False)

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats
        )
        subject.schedule_tasks()

        for trial_group in trial_groups:
            for metric in metrics:
                self.assertIn(mock.call(
                    trial_results=trial_group,
                    metric=metric,
                    num_cpus=mock.ANY, num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_measure_trial_task.call_args_list,
                    "Couldn't find measure_trial_task for metric {0}, trials {1}".format(
                        metric.pk, [t.pk for t in trial_group])
                )

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_stores_metric_results(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(3)]
        repeats = 2

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        make_mock_get_run_system_task(mock_task_manager, trial_results_map)

        metric_results = []

        def mock_get_measure_trial_task(trial_results, metric, *_, **__):
            result = mock_core.MockMetricResult(
                _id=ObjectId(),
                metric=metric,
                trial_results=trial_results,
                success=True
            )
            metric_results.append(result)
            mock_task = mock.MagicMock()
            mock_task.is_finished = True
            mock_task.result = result
            return mock_task

        mock_task_manager.get_measure_trial_task.side_effect = mock_get_measure_trial_task

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats,
        )
        subject.schedule_tasks()

        for metric_result in metric_results:
            self.assertIn(metric_result, subject.metric_results)

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_uses_measure_trial_tasks_correctly(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(1)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(1)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(1)]
        repeats = 1

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        make_mock_get_run_system_task(mock_task_manager, trial_results_map)

        metric_results = []

        def mock_get_measure_trial_task(trial_results, metric, *_, **__):
            result = mock_core.MockMetricResult(
                _id=ObjectId(),
                metric=metric,
                trial_results=trial_results,
                success=True
            )
            metric_results.append(result)
            mock_task = mock.create_autospec(MeasureTrialTask)
            mock_task.is_finished = True
            mock_task.result = result
            return mock_task

        mock_task_manager.get_measure_trial_task.side_effect = mock_get_measure_trial_task

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats,
        )
        subject.schedule_tasks()

        for metric_result in metric_results:
            self.assertIn(metric_result, subject.metric_results)


def make_trial_map(systems,  image_sources, repeats):
    """
    A helper to make trial results for each sytem for each image source for each repeat
    :param systems:
    :param image_sources:
    :param repeats:
    :return:
    """
    trial_results_map = {}
    for sys in systems:
        trial_results_map[sys.pk] = {}
        for img_source in image_sources:
            trial_results_map[sys.pk][img_source.pk] = []
            for repeat in range(repeats):
                trial_result = mock_core.MockTrialResult(
                    _id=ObjectId(),
                    image_source=img_source,
                    system=sys,
                    success=True
                )
                trial_results_map[sys.pk][img_source.pk].append(trial_result)
    return trial_results_map


def make_mock_get_run_system_task(mock_task_manager, trial_results_map, autospec=False):
    # Note - Autospeccing is expensive for every task, don't use by default
    def mock_get_run_system_task(system, image_source, repeat, *_, **__):
        if autospec:
            mock_task = mock.create_autospec(RunSystemTask)
        else:
            mock_task = mock.MagicMock()
        mock_task.is_finished = True
        mock_task.result = trial_results_map[system.pk][image_source.pk][repeat]
        return mock_task

    mock_task_manager.get_run_system_task.side_effect = mock_get_run_system_task
