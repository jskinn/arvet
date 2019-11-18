# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
from pymodm.errors import ValidationError
from bson import ObjectId
import arvet.database.tests.database_connection as dbconn
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.metric import Metric
import arvet.core.tests.mock_types as mock_core
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask
from arvet.batch_analysis.simple_experiment import SimpleExperiment


# ------------------------- HELPER TYPES -------------------------


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


class CountedMetricResult(mock_core.MockMetricResult):
    instances = 0

    def __init__(self, *args, **kwargs):
        super(CountedMetricResult, self).__init__(*args, **kwargs)
        self._inc_counter()

    @classmethod
    def _inc_counter(cls):
        cls.instances += 1


class CountedPlottedMetricResult1(CountedMetricResult):

    @classmethod
    def get_available_plots(cls):
        return {'my_plot_1'}


class CountedPlottedMetricResult2(CountedMetricResult):

    @classmethod
    def get_available_plots(cls):
        return {'plot_2', 'custom_demo_plot_newest'}


# ------------------------- DATABASE TESTS -------------------------


class TestExperimentDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def tearDown(self) -> None:
        # Clean up any objects we've created
        SimpleExperiment.objects.all().delete()
        mock_core.MockMetricResult.objects.all().delete()
        mock_core.MockTrialResult.objects.all().delete()
        mock_core.MockSystem.objects.all().delete()
        mock_core.MockImageSource.objects.all().delete()
        mock_core.MockMetric.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        SimpleExperiment._mongometa.collection.drop()
        mock_core.MockMetric._mongometa.collection.drop()
        mock_core.MockTrialResult._mongometa.collection.drop()
        mock_core.MockSystem._mongometa.collection.drop()
        mock_core.MockImageSource._mongometa.collection.drop()
        mock_core.MockMetricResult._mongometa.collection.drop()

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

    def test_stores_and_loads_minimal(self):
        obj = SimpleExperiment(name="TestSimpleExperiment")
        obj.save()

        # Load all the entities
        all_entities = list(SimpleExperiment.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)

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
            systems=[system1]
        )
        obj.save()

        del obj  # Clear the obj to clear existing objects and references
        CountedSystem.instances = 0

        obj = SimpleExperiment.objects.all().first()
        self.assertEqual(0, CountedSystem.instances)
        obj.add_vision_systems([system1, system2])
        self.assertEqual(0, CountedSystem.instances)
        # this will auto-dereference
        self.assertEqual(obj.systems, [system1, system2])

        # check we can still save
        obj.save()

    def test_add_vision_systems_enforces_uniqueness_if_already_dereferenced(self):
        system1 = mock_core.MockSystem()
        system1.save()

        system2 = mock_core.MockSystem()
        system2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=[system1]
        )
        obj.add_vision_systems([system1, system2])
        self.assertEqual(obj.systems, [system1, system2])

        # check we can still save
        obj.save()

    def test_add_image_sources_enforces_uniqueness_without_dereferencing(self):
        image_source1 = CountedImageSource()
        image_source1.save()

        image_source2 = CountedImageSource()
        image_source2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            image_sources=[image_source1]
        )
        obj.save()

        del obj  # Clear the obj to clear existing objects and references
        CountedImageSource.instances = 0

        obj = SimpleExperiment.objects.all().first()
        self.assertEqual(0, CountedImageSource.instances)
        obj.add_image_sources([image_source1, image_source2])
        self.assertEqual(0, CountedImageSource.instances)
        # this will auto-dereference
        self.assertEqual(obj.image_sources, [image_source1, image_source2])

        # check we can still save
        obj.save()

    def test_add_image_sources_enforces_uniqueness_if_already_dereferenced(self):
        image_source1 = mock_core.MockImageSource()
        image_source1.save()

        image_source2 = mock_core.MockImageSource()
        image_source2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            image_sources=[image_source1]
        )
        obj.add_image_sources([image_source1, image_source2])
        self.assertEqual(obj.image_sources, [image_source1, image_source2])

        # check we can still save
        obj.save()

    def test_add_metrics_enforces_uniqueness_without_dereferencing(self):
        metric1 = CountedMetric()
        metric1.save()

        metric2 = CountedMetric()
        metric2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            metrics=[metric1]
        )
        obj.save()

        del obj  # Clear the obj to clear existing objects and references
        CountedMetric.instances = 0

        obj = SimpleExperiment.objects.all().first()
        self.assertEqual(0, CountedMetric.instances)
        obj.add_metrics([metric1, metric2])
        self.assertEqual(0, CountedMetric.instances)
        # this will auto-dereference
        self.assertEqual(obj.metrics, [metric1, metric2])

        # check we can still save
        obj.save()

    def test_add_metrics_enforces_uniqueness_if_already_dereferenced(self):
        metric1 = mock_core.MockMetric()
        metric1.save()

        metric2 = mock_core.MockMetric()
        metric2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            metrics=[metric1]
        )
        obj.add_metrics([metric1, metric2])
        self.assertEqual(obj.metrics, [metric1, metric2])

        # check we can still save
        obj.save()

    def test_get_plots_returns_result_plots_without_dereferencing(self):
        # Build the heirarchy of references necessary to save a metric result
        system = mock_core.MockSystem()
        system.save()

        image_source = mock_core.MockImageSource()
        image_source.save()

        metric = mock_core.MockMetric()
        metric.save()

        trial_result = mock_core.MockTrialResult(system=system, image_source=image_source, success=True)
        trial_result.save()

        CountedPlottedMetricResult1.instances = 0
        CountedPlottedMetricResult2.instances = 0

        # Make some metric results that provide plots
        metric_result_1 = CountedPlottedMetricResult1(metric=metric, trial_results=[trial_result], success=True)
        metric_result_1.save()

        metric_result_2 = CountedPlottedMetricResult2(metric=metric, trial_results=[trial_result], success=True)
        metric_result_2.save()

        obj = SimpleExperiment(
            name="TestSimpleExperiment",
            metric_results=[metric_result_1, metric_result_2]
        )
        obj.save()

        # Check that creating the metric results increased the instance count
        self.assertEqual(1, CountedPlottedMetricResult1.instances)
        self.assertEqual(1, CountedPlottedMetricResult2.instances)

        del obj  # Clear the obj to clear existing objects and references
        CountedPlottedMetricResult1.instances = 0
        CountedPlottedMetricResult2.instances = 0

        obj = SimpleExperiment.objects.all().first()
        self.assertEqual(CountedPlottedMetricResult1.get_available_plots() |
                         CountedPlottedMetricResult2.get_available_plots(), obj.get_plots())
        self.assertEqual(0, CountedPlottedMetricResult1.instances)
        self.assertEqual(0, CountedPlottedMetricResult2.instances)


# ------------------------- TESTS WITHOUT DATABASE -------------------------


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

        for group_idx, trial_group in enumerate(trial_groups):
            for metric_idx, metric in enumerate(metrics):
                self.assertIn(mock.call(
                    trial_results=trial_group,
                    metric=metric,
                    num_cpus=mock.ANY, num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_measure_trial_task.call_args_list,
                    "\nCouldn't find measure_trial_task for metric {0}, trial group {1}".format(metric_idx, group_idx)
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
            mock_task.get_result.return_value = result
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
            mock_task.get_result.return_value = result
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

    def test_get_plots_returns_plots_from_metric_results(self):

        class PlottedMetricResult1(mock_core.MockMetricResult):

            @classmethod
            def get_available_plots(cls):
                return {'my_awesome_plot', 'plot_2_fixed'}

        class PlottedMetricResult2(mock_core.MockMetricResult):

            @classmethod
            def get_available_plots(cls):
                return {'demoplot_newer_2'}

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            metric_results=[PlottedMetricResult1(), PlottedMetricResult2()]
        )
        self.assertEqual({'my_awesome_plot', 'plot_2_fixed', 'demoplot_newer_2'}, subject.get_plots())

    def test_get_plots_requests_each_type_only_once(self):
        call_count = 0

        class PlotttedMetricResult(mock_core.MockMetricResult):

            @classmethod
            def get_available_plots(cls):
                nonlocal call_count
                call_count += 1
                return {'my_awesome_plot', 'plot_2'}

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            metric_results=[PlotttedMetricResult() for _ in range(10)]
        )
        self.assertEqual({'my_awesome_plot', 'plot_2'}, subject.get_plots())
        self.assertEqual(1, call_count)


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
        mock_task.get_result.return_value = trial_results_map[system.pk][image_source.pk][repeat]
        return mock_task

    mock_task_manager.get_run_system_task.side_effect = mock_get_run_system_task
