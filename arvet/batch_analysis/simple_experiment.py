import typing
import bson
import logging
from secrets import randbits
import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
from arvet.database.reference_list_field import ReferenceListField
from arvet.core.system import VisionSystem, StochasticBehaviour
from arvet.core.image_source import ImageSource
from arvet.core.metric import Metric
from arvet.batch_analysis.experiment import Experiment, run_all, run_all_with_seeds, measure_all


class SimpleExperiment(Experiment):
    """
    A subtype of experiment for simple cases where all systems are run with all datasets,
    and then are run with all benchmarks.
    If this common case is not your experiment, override base Experiment instead
    """
    systems = ReferenceListField(VisionSystem, on_delete=fields.ReferenceField.PULL, blank=True)
    image_sources = ReferenceListField(ImageSource, on_delete=fields.ReferenceField.PULL, blank=True)
    metrics = ReferenceListField(Metric, on_delete=fields.ReferenceField.PULL, blank=True)
    repeats = fields.IntegerField(required=True, default=1)
    use_seed = fields.BooleanField(required=True, default=False)
    seeds = fields.ListField(fields.IntegerField(), default=[])

    run_cpus = fields.IntegerField(default=1)
    run_gpus = fields.IntegerField(default=0)
    run_memory = fields.CharField(default='3GB')
    run_duration = fields.CharField(default='1:00:00')

    measure_cpus = fields.IntegerField(default=1)
    measure_gpus = fields.IntegerField(default=0)
    measure_memory = fields.CharField(default='3GB')
    measure_duration = fields.CharField(default='1:00:00')

    def schedule_tasks(self):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :return:
        """
        # Load the referenced models. We need this before we can dereference our reference fields
        self.load_referenced_models()
        self.clean_references()

        if len(self.image_sources) <= 0 or len(self.systems) <= 0:
            return

        if self.use_seed:
            # We're doing experiments with the seed, vary that
            if len(self.seeds) < self.repeats:
                # Choose any seeds we're missing
                # We need to save them so that we can get the same results next time
                self.seeds += [randbits(32) for _ in range(self.repeats - len(self.seeds))]
            trial_results, trials_remaining = run_all_with_seeds(
                systems=self.systems,
                image_sources=self.image_sources,
                seeds=self.seeds,
                num_cpus=self.run_cpus,
                num_gpus=self.run_gpus,
                memory_requirements=self.run_memory,
                expected_duration=self.run_duration
            )
        else:
            trial_results, trials_remaining = run_all(
                systems=self.systems,
                image_sources=self.image_sources,
                repeats=self.repeats,
                num_cpus=self.run_cpus,
                num_gpus=self.run_gpus,
                memory_requirements=self.run_memory,
                expected_duration=self.run_duration
            )

        if len(self.metrics) <= 0:
            return

        # A horrifying generator to pull out all the trial results that have enough repeats
        # Deterministic systems only need (or will get) a single repeat,
        # Other systems must have as many as the completed
        complete_groups = [
            trial_results[system.pk][image_source.pk]
            for system in self.systems
            for image_source in self.image_sources
            if system.pk in trial_results
            and image_source.pk in trial_results[system.pk]
            and (
                (
                    system.is_deterministic() is StochasticBehaviour.DETERMINISTIC and
                    len(trial_results[system.pk][image_source.pk]) >= 1
                ) or
                (
                    system.is_deterministic() is not StochasticBehaviour.DETERMINISTIC and
                    len(trial_results[system.pk][image_source.pk]) >= self.repeats
                )
            )
        ]
        metric_results, metrics_remaining = measure_all(
            metrics=self.metrics,
            trial_result_groups=complete_groups,
            num_cpus=self.measure_cpus,
            num_gpus=self.measure_gpus,
            memory_requirements=self.measure_memory,
            expected_duration=self.measure_duration
        )
        self.metric_results.extend(
            metric_result for metric_result_list in metric_results.values()
            for metric_result in metric_result_list
        )

    def load_referenced_models(self):
        """
        Go through the models referenced by this experiment and ensure their model types have been loaded.
        This is necessary or accessing any of the reference fields will cause an exception
        :return:
        """
        # Load system models
        with no_auto_dereference(type(self)):
            model_ids = set(sys_id for sys_id in self.systems if isinstance(sys_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(VisionSystem, ids=list(model_ids))

        # Load image source models
        with no_auto_dereference(type(self)):
            model_ids = set(source_id for source_id in self.image_sources if isinstance(source_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(ImageSource, ids=list(model_ids))

        # Load metric models
        with no_auto_dereference(type(self)):
            model_ids = set(metric_id for metric_id in self.metrics if isinstance(metric_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(Metric, ids=list(model_ids))

    def clean_references(self):
        """
        Clean up our lists of systems, image sources, and metrics.
        This fixes cases where a system, image source, or metric has been pulled from the list.
        The experiment may fail to save with nulls in its reference lists.
        :return:
        """
        super(SimpleExperiment, self).clean_references()
        self.systems = [system for system in self.systems if system is not None]
        self.image_sources = [image_source for image_source in self.image_sources if image_source is not None]
        self.metrics = [metric for metric in self.metrics if metric is not None]

    def add_vision_systems(self, vision_systems: typing.Iterable[VisionSystem]):
        """
        Add the given vision systems to this experiment if they are not already associated with it
        :param vision_systems:
        :return:
        """
        with no_auto_dereference(type(self)):
            if self.systems is None:
                existing_pks = set()
            else:
                existing_pks = {system.pk if hasattr(system, 'pk') else system for system in self.systems}
            new_vision_systems = [vision_system for vision_system in vision_systems
                                  if vision_system.pk not in existing_pks]
            if len(new_vision_systems) > 0:
                if self.systems is None:
                    self.systems = new_vision_systems
                else:
                    self.systems.extend(new_vision_systems)

    def add_image_sources(self, image_sources: typing.Iterable[ImageSource]):
        """
        Add the given image sources to this experiment if they are not already associated with it
        :param image_sources:
        :return:
        """
        with no_auto_dereference(type(self)):
            if self.image_sources is None:
                existing_pks = set()
            else:
                existing_pks = {image_source.pk if hasattr(image_source, 'pk') else image_source
                                for image_source in self.image_sources}
            new_image_sources = [image_source for image_source in image_sources if image_source.pk not in existing_pks]
            if len(new_image_sources) > 0:
                if self.image_sources is None:
                    self.image_sources = new_image_sources
                else:
                    self.image_sources.extend(new_image_sources)

    def add_metrics(self, metrics: typing.Iterable[Metric]):
        """
        Add the given metrics to this experiment if they are not already associated with it
        :param metrics:
        :return:
        """
        with no_auto_dereference(type(self)):
            if self.metrics is None:
                existing_pks = set()
            else:
                existing_pks = {metric.pk if hasattr(metric, 'pk') else metric for metric in self.metrics}
            new_metrics = [metric for metric in metrics if metric.pk not in existing_pks]
            if len(new_metrics) > 0:
                if self.metrics is None:
                    self.metrics = new_metrics
                else:
                    self.metrics.extend(new_metrics)
        pass
