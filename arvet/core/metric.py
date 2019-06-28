# Copyright (c) 2017, John Skinner
import abc
import typing
import bson
import pandas as pd
import pymodm
import pymodm.fields as fields
import arvet.database.pymodm_abc as pymodm_abc
from arvet.database.reference_list_field import ReferenceListField
import arvet.core.trial_result


class Metric(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    A class that measures results

    This is an abstract base class defining an interface for all metrics,
    to allow them to be called easily and in a structured way.
    """

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the id for this metric
        :return:
        """
        return self._id

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

    @classmethod
    def get_pretty_name(cls) -> str:
        """
        Get a human-readable name for this metric
        :return:
        """
        return cls.__module__ + '.' + cls.__name__

    @classmethod
    def get_instance(cls) -> 'Metric':
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


class MetricResult(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    A general superclass for metric results for all metrics
    """
    metric = fields.ReferenceField(Metric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results = ReferenceListField(arvet.core.trial_result.TrialResult,
                                       required=True, on_delete=fields.ReferenceField.CASCADE)
    success = fields.BooleanField(required=True)
    message = fields.CharField()

    # The set of plots available to visualize_results.
    available_plots = set()

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the id of this metric result
        :return:
        """
        return self._id

    @abc.abstractmethod
    def get_columns(self) -> typing.Set[str]:
        """
        Get a list of available results columns
        Should delegate to the linked trial results, systems, etc for the full list
        :return:
        """
        pass

    @abc.abstractmethod
    def get_results(self, columns: typing.Iterable[str] = None) -> typing.List[dict]:
        """

        :return:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def visualize_results(cls, results: typing.Iterable['MetricResult'], output_folder: str,
                          plots: typing.Iterable[str] = None) -> None:
        """

        :param results:
        :param plots:
        :param output_folder:
        :return:
        """
        pass


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
