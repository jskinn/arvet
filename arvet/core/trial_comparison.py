# Copyright (c) 2017, John Skinner
import abc
import bson
import pymodm
import pymodm.fields as fields
import typing
import arvet.database.pymodm_abc as pymodm_abc
from arvet.database.reference_list_field import ReferenceListField
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

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the id of this trial comparison metric
        :return:
        """
        return self._id

    @abc.abstractmethod
    def is_trial_appropriate_for_first(self, trial_result: arvet.core.trial_result.TrialResult) -> bool:
        """
        More fine-grained filtering for trial results,
        to make sure this class can benchmark this trial result.
        :return: 
        """
        pass

    @abc.abstractmethod
    def is_trial_appropriate_for_second(self, trial_result: arvet.core.trial_result.TrialResult) -> bool:
        """
        More fine-grained filtering for trial results,
        to make sure this class can benchmark this trial result.
        :return:
        """
        pass

    @abc.abstractmethod
    def compare_trials(self, trial_results_1: typing.Iterable[arvet.core.trial_result.TrialResult],
                       trial_results_2: typing.Iterable[arvet.core.trial_result.TrialResult]) -> \
            MetricResult:
        """
        Compare the results of the first group of trials with a group of reference trials.

        :param trial_results_1: An iterable of TrialResults to compare
        :param trial_results_2: An iterable of TrialResult to compare to
        :return: A MetricResult object containing either the results, or explaining the error
        :rtype: MetricResult
        """
        pass

    @classmethod
    def get_pretty_name(cls) -> str:
        """
        Get a human-readable name for this metric
        :return:
        """
        return cls.__module__ + '.' + cls.__name__

    @classmethod
    def get_instance(cls) -> 'TrialComparisonMetric':
        """
        Get an instance of this vision system, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        all_objects = cls.objects.all()
        if all_objects.count() > 0:
            return all_objects.first()
        obj = cls()
        return obj


class TrialComparisonResult(pymodm.MongoModel):
    """
    A general superclass for metric results that compare two groups trials.
    """
    metric = fields.ReferenceField(TrialComparisonMetric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results_1 = ReferenceListField(arvet.core.trial_result.TrialResult,
                                         required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results_2 = ReferenceListField(arvet.core.trial_result.TrialResult,
                                         required=True, on_delete=fields.ReferenceField.CASCADE)
    success = fields.BooleanField(required=True)
    message = fields.CharField()

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the id of this trial comparison result
        :return:
        """
        return self._id
