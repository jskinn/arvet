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

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name used to refer to this metric
        :return:
        """
        return cls.__module__ + '.' + cls.__name__

    @abc.abstractmethod
    def is_trial_appropriate(self, trial_result: arvet.core.trial_result.TrialResult) -> bool:
        """
        Fine-grained filtering for trial results, to make sure this class can benchmark this trial result.
        :return:
        """
        pass

    @abc.abstractmethod
    def benchmark_results(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) \
            -> 'MetricResult':
        """
        Benchmark the results of running a particular system on a particular image source.
        We take a collection of trials to allow for multiple repeats of the system on the same data,
        which allows us to account for and measure random variation in the system.
        A helper to check this is provided below, call it in any implementation.

        The trial result MUST include the ground truth along with the system estimates,
        which must be the same for all trials.

        :param trial_results: A collection of trial results to benchmark.
        These are assumed to be repeat runs of the same system on the same data.
        :return: A MetricResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: MetricResult
        """
        pass


class MetricResult(pymodm.MongoModel):
    """
    A general superclass for benchmark results for all benchmarks
    """
    metric = fields.ReferenceField(Metric, required=True, on_delete=fields.ReferenceField.CASCADE)
    trial_results = fields.ListField(fields.ReferenceField(arvet.core.trial_result.TrialResult,
                                                           required=True, on_delete=fields.ReferenceField.CASCADE))
    success = fields.BooleanField(required=True)
    message = fields.CharField()


def check_trial_collection(trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) \
        -> typing.Union[str, None]:
    """
    A helper function to check that all the given trial results come from the same system and image source.
    Call this at the start of Benchmark.benchmark_results and return the result as a FailedBenchmark if it is not None
    TODO: We are not tracking which image source a trial result comes from, sort that out.
    :param trial_results: A collection of trial results passed to Benchmark.benchmark_results
    :return: None if all the trials are OK, string explaining the problem if they are not
    """
    system_ids = set(trial_result.system_id for trial_result in trial_results)
    if len(system_ids) > 1:
        # There are two different systems, produce the error message:
        return "Given trial results contain results from multiple systems: {0}".format(system_ids)
    return None
