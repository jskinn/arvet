# Copyright (c) 2017, John Skinner
import logging
import pymodm.fields as fields
from arvet.config.path_manager import PathManager
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
import arvet.batch_analysis.task


class MeasureTrialTask(arvet.batch_analysis.task.Task):
    """
    A task for benchmarking a trial result. Result is a BenchmarkResult id.
    """
    metric = fields.ReferenceField(Metric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results = fields.ListField(
        fields.ReferenceField(TrialResult, required=True, on_delete=fields.ReferenceField.CASCADE),
        required=True)
    result = fields.ReferenceField(MetricResult, on_delete=fields.ReferenceField.CASCADE)

    def run_task(self, path_manager: PathManager) -> None:
        import traceback

        # Check all the trials are appropriate
        for trial_result in self.trial_results:
            if not self.metric.is_trial_appropriate(trial_result):
                # Metric cannot measure these trials, fail permanently
                self.fail_with_message("Metric {0} cannot assess trial".format(self.metric.get_pretty_name()))
                return

        logging.getLogger(__name__).info(
            "Running metric {0}".format(self.metric.get_pretty_name()))
        try:
            metric_result = self.metric.measure_results(self.trial_results)
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
                "Measured trials using metric {0}, but got unsuccessful result: {1}".format(
                    self.metric.get_pretty_name(), metric_result.message))
        else:
            logging.getLogger(__name__).info(
                "Successfully measured trials using metric {0}".format(self.metric.get_pretty_name()))
        self.result = metric_result
        self.mark_job_complete()

    def fail_with_message(self, message):
        """
        Quick helper to log error message, and make and store a metric result as the result
        :param message:
        :return:
        """
        logging.getLogger(__name__).error(message)
        self.result = MetricResult(
            metric=self.metric,
            trial_results=self.trial_results,
            success=False,
            message=message
        )
        self.mark_job_complete()
