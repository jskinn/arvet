import pymodm.fields as fields
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
