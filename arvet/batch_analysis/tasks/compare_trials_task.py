# Copyright (c) 2017, John Skinner
import logging
import pymodm.fields as fields
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
