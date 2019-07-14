# Copyright (c) 2017, John Skinner
import logging
import typing
import bson
import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
from arvet.core.trial_result import TrialResult
from arvet.core.trial_comparison import TrialComparisonMetric, TrialComparisonResult
import arvet.batch_analysis.task
from arvet.config.path_manager import PathManager


class CompareTrialTask(arvet.batch_analysis.task.Task):
    """
    A task for comparing two trial results against each other. Result is a TrialComparison id
    """

    metric = fields.ReferenceField(TrialComparisonMetric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results_1 = fields.ListField(
        fields.ReferenceField(TrialResult, required=True, on_delete=fields.ReferenceField.CASCADE),
        required=True)
    trial_results_2 = fields.ListField(
        fields.ReferenceField(TrialResult, required=True, on_delete=fields.ReferenceField.CASCADE),
        required=True)
    result = fields.ReferenceField(TrialComparisonResult, on_delete=fields.ReferenceField.CASCADE)

    def run_task(self, path_manager: PathManager) -> None:
        import traceback

        # Load all the referenced models
        self.load_referenced_modules()

        # Check all the trials are appropriate
        for trial_num, trial_result in enumerate(self.trial_results_1):
            if not self.metric.is_trial_appropriate_for_first(trial_result):
                # Metric cannot measure these trials, fail permanently
                self.fail_with_message("Comparison metric {0} cannot assess trial {1} in group 1".format(
                    self.metric.get_pretty_name(), trial_num))
                return
        for trial_num, trial_result in enumerate(self.trial_results_2):
            if not self.metric.is_trial_appropriate_for_second(trial_result):
                # Metric cannot measure these trials, fail permanently
                self.fail_with_message("Comparison metric {0} cannot assess trial {1} in group 2".format(
                    self.metric.get_pretty_name(), trial_num))
                return

        logging.getLogger(__name__).info(
            "Running comparison metric {0}".format(self.metric.get_pretty_name()))
        try:
            metric_result = self.metric.compare_trials(self.trial_results_1, self.trial_results_2)
        except Exception as exception:
            self.fail_with_message("Exception while running metric {0}:\n{1}".format(
                self.metric.get_pretty_name(), traceback.format_exc()))
            raise exception  # Re-raise the caught exception
        if metric_result is None:
            self.fail_with_message("Failed to run {0}, metric returned None".format(
                self.metric.get_pretty_name()))
            return

        if not metric_result.success:
            logging.getLogger(__name__).info(
                "Compared trials using metric {0}, but got unsuccessful result: {1}".format(
                    self.metric.get_pretty_name(), metric_result.message))
        else:
            logging.getLogger(__name__).info(
                "Successfully Compared trials using metric {0}".format(self.metric.get_pretty_name()))
        self.result = metric_result
        self.mark_job_complete()

    def load_referenced_modules(self):
        logging.getLogger(__name__).info("Loading referenced models...")
        # Load the metric model
        metric_id = None
        with no_auto_dereference(CompareTrialTask):
            if isinstance(self.metric, bson.ObjectId):
                metric_id = self.metric
        if metric_id is not None:
            autoload_modules(TrialComparisonMetric, [metric_id])

        # Load the trial results models
        with no_auto_dereference(CompareTrialTask):
            model_ids = list(
                set(tr_id for tr_id in self.trial_results_1 if isinstance(tr_id, bson.ObjectId)) |
                set(tr_id for tr_id in self.trial_results_2 if isinstance(tr_id, bson.ObjectId))
            )
        if len(model_ids) > 0:
            autoload_modules(TrialResult, model_ids)

    def fail_with_message(self, message):
        """
        Quick helper to log error message, and make and store a metric result as the result
        :param message:
        :return:
        """
        logging.getLogger(__name__).error(message)
        self.result = TrialComparisonResult(
            metric=self.metric,
            trial_results_1=self.trial_results_1,
            trial_results_2=self.trial_results_2,
            success=False,
            message=message
        )
        self.mark_job_complete()

    @property
    def result_id(self) -> typing.Union[bson.ObjectId, None]:
        """
        Get the id of the result, without attempting to construct the object.
        Makes it easier for other objects to refer to this result, without loading large result objects.
        :return:
        """
        with no_auto_dereference(CompareTrialTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                return self.result
        return self.result.pk

    def get_result(self) -> typing.Union[TrialComparisonResult, None]:
        """
        Actually get the result object.
        This will auto-load the result model, and then attempt to construct it.
        :return:
        """
        with no_auto_dereference(CompareTrialTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                # result is an id and not a model, autoload the model
                autoload_modules(TrialComparisonResult, [self.result])
        # This will now dereference correctly
        return self.result
