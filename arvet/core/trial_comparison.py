# Copyright (c) 2017, John Skinner
import abc
import pymodm
import pymodm.fields as fields
import typing
import arvet.database.pymodm_abc as pymodm_abc
import arvet.core.trial_result
from arvet.core.metric import MetricResult


class TrialComparisonMetric(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    Some metrics and performance measures only make sense when comparing two different sets of trial results,
    that is, by comparing two similar runs and measuring the difference in some way.
    Metrics of this type take two different groups of trial results, and measures the difference between them.

    This is an abstract base class defining an interface for all such benchmarks,
    to allow them to be called easily and in a structured way.
    """

    @abc.abstractmethod
    def is_trial_appropriate(self, trial_result: arvet.core.trial_result.TrialResult) -> bool:
        """
        More fine-grained filtering for trial results,
        to make sure this class can benchmark this trial result.
        :return: 
        """
        pass

    @abc.abstractmethod
    def compare_trials(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult],
                       reference_trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) -> \
            MetricResult:
        """
        Compare the results of the first group of trials with a group of reference trials.

        :param trial_results: An iterable of TrialResults to compare
        :param reference_trial_results: An iterable of TrialResult to compare to
        :return: A MetricResult object containing either the results, or explaining the error
        :rtype: MetricResult
        """
        pass


class TrialComparisonResult(MetricResult):
    """
    A general superclass for metric results that compare two groups trials.
    """
    reference_trial_results = fields.ListField(
        fields.ReferenceField(arvet.core.trial_result.TrialResult,
                              required=True, on_delete=fields.ReferenceField.CASCADE))
