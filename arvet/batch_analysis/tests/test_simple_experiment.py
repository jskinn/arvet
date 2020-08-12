# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
from pymodm.errors import ValidationError
from pymodm.context_managers import no_auto_dereference
from bson import ObjectId
import arvet.database.tests.database_connection as dbconn
from arvet.core.system import VisionSystem, StochasticBehaviour
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


# ------------------------- DATABASE TESTS -------------------------


class TestSimpleExperimentDatabase(unittest.TestCase):

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
            metrics=[metric],
            plots=[]
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
        self.assertIn(mock.call(VisionSystem, ids=[system.pk]), mock_autoload.call_args_list)
        self.assertIn(mock.call(ImageSource, ids=[image_source.pk]), mock_autoload.call_args_list)
        self.assertIn(mock.call(Metric, ids=[metric.pk]), mock_autoload.call_args_list)

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


# ------------------------- TESTS WITHOUT DATABASE -------------------------


class TestSimpleExperiment(unittest.TestCase):

    def test_refill_seeds_makes_a_unique_set_equal_to_the_number_of_repeats(self):
        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            repeats=20,
            use_seed=True
        )
        subject.refill_seeds()
        self.assertEqual(20, len(subject.seeds))
        self.assertEqual(len(set(subject.seeds)), len(subject.seeds))

    def test_refill_seeds_makes_a_different_set_every_time(self):
        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            repeats=20,
            use_seed=True
        )
        subject.refill_seeds()
        seeds_1 = set(subject.seeds)
        subject.seeds = []
        subject.refill_seeds()
        self.assertNotEqual(seeds_1, set(subject.seeds))

    def test_refill_seeds_preserves_existing_seeds(self):
        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            repeats=20,
            use_seed=True,
            seeds=[1, 2, 3]
        )
        subject.refill_seeds()
        self.assertIn(1, subject.seeds)
        self.assertIn(2, subject.seeds)
        self.assertIn(3, subject.seeds)

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_runs_all_systems_with_all_image_sources_repeatedly(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        systems[0].is_deterministic = mock.create_autospec(systems[0].is_deterministic,
                                                           return_value=StochasticBehaviour.NON_DETERMINISTIC)
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metric = mock_core.MockMetric(_id=ObjectId())
        repeats = 2

        run_cpus = 3
        run_gpus = 2
        run_memory = '9GB'
        run_duration = '2:33:44'

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=[metric],
            repeats=repeats,
            run_cpus=run_cpus,
            run_gpus=run_gpus,
            run_memory=run_memory,
            run_duration=run_duration
        )
        subject.schedule_tasks()

        for system in systems:
            for image_source in image_sources:
                actual_repeats = repeats if system.is_deterministic() is not StochasticBehaviour.DETERMINISTIC else 1
                for repeat in range(actual_repeats):
                    self.assertIn(mock.call(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        num_cpus=run_cpus, num_gpus=run_gpus,
                        memory_requirements=run_memory,
                        expected_duration=run_duration
                    ), mock_task_manager.get_run_system_task.call_args_list)

    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_runs_all_systems_with_all_image_sources_with_all_seeds(self, mock_task_manager):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        systems[0].is_deterministic = mock.create_autospec(systems[0].is_deterministic,
                                                           return_value=StochasticBehaviour.SEEDED)
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metric = mock_core.MockMetric(_id=ObjectId())
        repeats = 2

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=[metric],
            repeats=repeats,
            use_seed=True
        )
        subject.schedule_tasks()
        self.assertEqual(repeats, len(subject.seeds))   # The set of seeds should have auto-filled
        self.assertEqual(len(subject.seeds), len(set(subject.seeds)))   # The set of seeds must be distinct

        for system in systems:
            for image_source in image_sources:
                for seed in subject.seeds:
                    self.assertIn(mock.call(
                        system=system,
                        image_source=image_source,
                        repeat=0,
                        seed=seed,
                        num_cpus=mock.ANY, num_gpus=mock.ANY,
                        memory_requirements=mock.ANY,
                        expected_duration=mock.ANY
                    ), mock_task_manager.get_run_system_task.call_args_list)

    @mock.patch('arvet.batch_analysis.experiment.load_minimal_trial_result', autospec=True)
    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_uses_run_system_tasks_correctly(self, mock_task_manager, mock_load_minimal_trial):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(1)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(1)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(1)]
        repeats = 1

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        # Note: This is a separate test because autospecs are expensive in a loop
        make_mock_get_run_system_task(mock_task_manager, trial_results_map, autospec=True)
        patch_load_minimal_trial(mock_load_minimal_trial)

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats
        )
        subject.schedule_tasks()

    @mock.patch('arvet.batch_analysis.experiment.load_minimal_trial_result', autospec=True)
    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_measures_all_trial_results(self, mock_task_manager, mock_load_minimal_trial):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        systems[0].is_deterministic = mock.create_autospec(systems[0].is_deterministic,
                                                           return_value=StochasticBehaviour.NON_DETERMINISTIC)
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(3)]
        repeats = 3

        measure_cpus = 16
        measure_gpus = 22
        measure_memory = '88GB'
        measure_duration = '0:01:01'

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        trial_groups = [
            trial_group
            for trials_by_source in trial_results_map.values()
            for trial_group in trials_by_source.values()
        ]
        trials_by_id = {
            trial_result.pk: trial_result
            for trial_group in trial_groups
            for trial_result in trial_group
        }
        make_mock_get_run_system_task(mock_task_manager, trial_results_map, autospec=False)
        patch_load_minimal_trial(mock_load_minimal_trial)

        def mock_load_minimal_trial_impl(object_id):
            if object_id in trials_by_id:
                return trials_by_id[object_id]
            raise ValueError(f"Unidentified trial result id {object_id}")

        mock_load_minimal_trial.side_effect = mock_load_minimal_trial_impl

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats,
            measure_cpus=measure_cpus,
            measure_gpus=measure_gpus,
            measure_memory=measure_memory,
            measure_duration=measure_duration
        )
        subject.schedule_tasks()

        for group_idx, trial_group in enumerate(trial_groups):
            for metric_idx, metric in enumerate(metrics):
                self.assertIn(mock.call(
                    trial_results=trial_group,
                    metric=metric,
                    num_cpus=measure_cpus, num_gpus=measure_gpus,
                    memory_requirements=measure_memory,
                    expected_duration=measure_duration
                ), mock_task_manager.get_measure_trial_task.call_args_list,
                    "\nCouldn't find measure_trial_task for metric {0}, trial group {1}".format(metric_idx, group_idx)
                )

    @mock.patch('arvet.batch_analysis.experiment.load_minimal_trial_result', autospec=True)
    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_stores_metric_results(self, mock_task_manager, mock_load_minimal_trial):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        systems[0].is_deterministic = mock.create_autospec(systems[0].is_deterministic,
                                                           return_value=StochasticBehaviour.NON_DETERMINISTIC)
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(3)]
        repeats = 2

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        make_mock_get_run_system_task(mock_task_manager, trial_results_map)
        patch_load_minimal_trial(mock_load_minimal_trial)

        metric_result_ids = []

        def mock_get_measure_trial_task(*_, **__):
            result_id = ObjectId()
            metric_result_ids.append(result_id)
            mock_task = mock.MagicMock()
            mock_task.is_finished = True
            mock_task.result_id = result_id
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

        with no_auto_dereference(SimpleExperiment):
            for metric_result_id in metric_result_ids:
                self.assertIn(metric_result_id, subject.metric_results)

    @mock.patch('arvet.batch_analysis.experiment.load_minimal_trial_result', autospec=True)
    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_discards_old_metric_results(self, mock_task_manager, mock_load_minimal_trial):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(3)]
        systems[0].is_deterministic = mock.create_autospec(systems[0].is_deterministic,
                                                           return_value=StochasticBehaviour.NON_DETERMINISTIC)
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(3)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(3)]
        repeats = 2

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        make_mock_get_run_system_task(mock_task_manager, trial_results_map)
        patch_load_minimal_trial(mock_load_minimal_trial)

        old_metric_results = [
            mock_core.MockMetricResult(_id=ObjectId(), success=True)
        ]
        metric_result_ids = []

        def mock_get_measure_trial_task(*_, **__):
            result_id = ObjectId()
            metric_result_ids.append(result_id)
            mock_task = mock.MagicMock()
            mock_task.is_finished = True
            mock_task.result_id = result_id
            return mock_task

        mock_task_manager.get_measure_trial_task.side_effect = mock_get_measure_trial_task

        subject = SimpleExperiment(
            name="TestSimpleExperiment",
            systems=systems,
            image_sources=image_sources,
            metrics=metrics,
            repeats=repeats,
            metric_results=old_metric_results
        )
        subject.schedule_tasks()

        with no_auto_dereference(SimpleExperiment):
            # Results should be all ids, from the measure_all
            for metric_result_id in metric_result_ids:
                self.assertIn(metric_result_id, subject.metric_results)
            for metric_result in old_metric_results:
                self.assertNotIn(metric_result.pk, subject.metric_results)

    @mock.patch('arvet.batch_analysis.experiment.load_minimal_trial_result', autospec=True)
    @mock.patch('arvet.batch_analysis.experiment.task_manager', autospec=True)
    def test_schedule_tasks_uses_measure_trial_tasks_correctly(self, mock_task_manager, mock_load_minimal_trial):
        systems = [mock_core.MockSystem(_id=ObjectId()) for _ in range(1)]
        image_sources = [mock_core.MockImageSource(_id=ObjectId()) for _ in range(1)]
        metrics = [mock_core.MockMetric(_id=ObjectId()) for _ in range(1)]
        repeats = 1

        trial_results_map = make_trial_map(systems, image_sources, repeats)
        make_mock_get_run_system_task(mock_task_manager, trial_results_map)
        patch_load_minimal_trial(mock_load_minimal_trial)

        metric_result_ids = []

        def mock_get_measure_trial_task(*_, **__):
            result_id = ObjectId()
            metric_result_ids.append(result_id)
            mock_task = mock.create_autospec(MeasureTrialTask)
            mock_task.is_finished = True
            mock_task.result_id = result_id
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

        with no_auto_dereference(SimpleExperiment):
            for metric_result_id in metric_result_ids:
                self.assertIn(metric_result_id, subject.metric_results)


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
            actual_repeats = repeats if sys.is_deterministic() is not StochasticBehaviour.DETERMINISTIC else 1
            for repeat in range(actual_repeats):
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
        mock_task.result_id = trial_results_map[system.pk][image_source.pk][repeat].pk
        return mock_task

    mock_task_manager.get_run_system_task.side_effect = mock_get_run_system_task


def patch_load_minimal_trial(mock_load_minimal_trial):
    def mock_load_minimal_trial_impl(object_id):
        mock_trial = mock.Mock()
        mock_trial.pk = object_id
        return mock_trial

    mock_load_minimal_trial.side_effect = mock_load_minimal_trial_impl
