import typing
import bson
import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
from arvet.database.reference_list_field import ReferenceListField
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.metric import Metric
import arvet.batch_analysis.experiment as ex


class SimpleExperiment(ex.Experiment):
    """
    A subtype of experiment for simple cases where all systems are run with all datasets,
    and then are run with all benchmarks.
    If this common case is not your experiment, override base Experiment instead
    """
    systems = ReferenceListField(VisionSystem, on_delete=fields.ReferenceField.PULL)
    image_sources = ReferenceListField(ImageSource, on_delete=fields.ReferenceField.PULL)
    metrics = ReferenceListField(Metric, on_delete=fields.ReferenceField.PULL)
    repeats = fields.IntegerField(required=True, default=1)

    def schedule_tasks(self):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :return:
        """
        # Load the referenced models. We need this before we can dereference our reference fields
        self.load_referenced_models()

        trial_results, trials_remaining = ex.run_all(
            systems=self.systems,
            image_sources=self.image_sources,
            repeats=self.repeats
        )
        complete_groups = [
            trial_result_group
            for trials_by_source in trial_results.values()
            for trial_result_group in trials_by_source.values()
            if len(trial_result_group) >= self.repeats
        ]
        metric_results, metrics_remaining = ex.measure_all(
            metrics=self.metrics,
            trial_result_groups=complete_groups
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
        with no_auto_dereference(SimpleExperiment):
            model_ids = set(sys_id for sys_id in self.systems if isinstance(sys_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(VisionSystem, ids=list(model_ids))

        # Load image source models
        with no_auto_dereference(SimpleExperiment):
            model_ids = set(source_id for source_id in self.image_sources if isinstance(source_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(ImageSource, ids=list(model_ids))

        # Load metric models
        with no_auto_dereference(SimpleExperiment):
            model_ids = set(metric_id for metric_id in self.metrics if isinstance(metric_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(Metric, ids=list(model_ids))

    def add_vision_systems(self, vision_systems: typing.Iterable[VisionSystem]):
        """
        Add the given vision systems to this experiment if they are not already associated with it
        :param vision_systems:
        :return:
        """
        with no_auto_dereference(SimpleExperiment):
            if self.systems is None:
                existing_pks = set()
            else:
                existing_pks = set(self.systems)
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
        with no_auto_dereference(SimpleExperiment):
            if self.image_sources is None:
                existing_pks = set()
            else:
                existing_pks = set(self.image_sources)
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
        with no_auto_dereference(SimpleExperiment):
            if self.metrics is None:
                existing_pks = set()
            else:
                existing_pks = set(self.metrics)
            new_metrics = [metric for metric in metrics if metric.pk not in existing_pks]
            if len(new_metrics) > 0:
                if self.metrics is None:
                    self.metrics = new_metrics
                else:
                    self.metrics.extend(new_metrics)
        pass
