# Copyright (c) 2017, John Skinner
import abc
import typing
import pymodm
import pymodm.fields as fields
import arvet.database.pymodm_abc as pymodm_abc
import arvet.core.trial_result


class Metric(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    A class that measures results

    This is an abstract base class defining an interface for all metrics,
    to allow them to be called easily and in a structured way.
    """

    @abc.abstractmethod
    def is_trial_appropriate(self, trial_result: arvet.core.trial_result.TrialResult) -> bool:
        """
        Fine-grained filtering for trial results, to make sure this class can measure this trial result.
        :return:
        """
        pass

    @abc.abstractmethod
    def measure_results(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) \
            -> 'MetricResult':
        """
        Measure the results of running a particular system on a particular image source.
        We take a collection of trials to allow for multiple repeats of the system on the same data,
        which allows us to account for and measure random variation in the system.
        A helper to check this is provided below, call it in any implementation.

        The trial result MUST include the ground truth along with the system estimates,
        which must be the same for all trials.

        :param trial_results: A collection of trial results to measure.
        These are assumed to be repeat runs of the same system on the same data.
        :return: A MetricResult object containing either the results, or explaining the error
        :rtype: MetricResult
        """
        pass


class MetricResult(pymodm.MongoModel):
    """
    A general superclass for metric results for all metrics
    """
    metric = fields.ReferenceField(Metric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results = fields.ListField(fields.ReferenceField(arvet.core.trial_result.TrialResult,
                                                           required=True, on_delete=fields.ReferenceField.CASCADE))
    success = fields.BooleanField(required=True)
    message = fields.CharField()


def check_trial_collection(trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) -> bool:
    """
    A helper function to check that all the given trial results come from the same system and image source.
    Call this at the start of Metric.measure_results
    :param trial_results: A collection of trial results passed to Metric.measure_results
    :return: None if all the trials are OK, string explaining the problem if they are not
    """
    first_trial = None
    for trial in trial_results:
        if first_trial is None:
            first_trial = trial
        elif trial.image_source != first_trial.image_source or trial.system != first_trial.system:
            return False
    return True
