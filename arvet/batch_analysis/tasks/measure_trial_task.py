# Copyright (c) 2017, John Skinner
import logging
import typing
import bson

import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
from arvet.database.reference_list_field import ReferenceListField
from arvet.config.path_manager import PathManager
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
import arvet.batch_analysis.task


class MeasureTrialTask(arvet.batch_analysis.task.Task):
    """
    A task for benchmarking a trial result. Result is a BenchmarkResult id.
    """
    metric = fields.ReferenceField(Metric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results = ReferenceListField(TrialResult, required=True, on_delete=fields.ReferenceField.CASCADE)
    result = fields.ReferenceField(MetricResult, on_delete=fields.ReferenceField.CASCADE)

    @property
    def result_id(self) -> typing.Union[bson.ObjectId, None]:
        """
        Get the id of the result, without attempting to construct the object.
        Makes it easier for other objects to refer to this result, without loading large result objects.
        :return:
        """
        with no_auto_dereference(MeasureTrialTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                return self.result
        return self.result.pk

    def get_result(self) -> typing.Union[MetricResult, None]:
        """
        Actually get the result object.
        This will auto-load the result model, and then attempt to construct it.
        :return:
        """
        with no_auto_dereference(MeasureTrialTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                # result is an id and not a model, autoload the model
                autoload_modules(MetricResult, [self.result])
        # This will now dereference correctly
        return self.result

    def run_task(self, path_manager: PathManager) -> None:
        import traceback

        # Load all the referenced models
        self.load_referenced_modules()

        # Check all the trials are appropriate
        for trial_num, trial_result in enumerate(self.trial_results):
            if not self.metric.is_trial_appropriate(trial_result):
                # Metric cannot measure these trials, fail permanently
                self.fail_with_message("Metric {0} cannot assess trial {1}".format(
                    self.metric.get_pretty_name(), trial_num))
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

    def load_referenced_modules(self):
        logging.getLogger(__name__).info("Loading referenced models...")
        # Load the metric model
        metric_id = None
        with no_auto_dereference(MeasureTrialTask):
            if isinstance(self.metric, bson.ObjectId):
                metric_id = self.metric
        if metric_id is not None:
            autoload_modules(Metric, [metric_id])

        # Load the trial results models
        with no_auto_dereference(MeasureTrialTask):
            model_ids = list(set(tr_id for tr_id in self.trial_results if isinstance(tr_id, bson.ObjectId)))
        if len(model_ids) > 0:
            autoload_modules(TrialResult, model_ids)

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
