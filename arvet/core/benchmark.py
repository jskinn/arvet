# Copyright (c) 2017, John Skinner
import abc
import bson
import typing
import arvet.database.entity
import arvet.core.trial_result


class Benchmark(arvet.database.entity.Entity, metaclass=abc.ABCMeta):
    """
    A class that benchmarks algorithms.

    This is an abstract base class defining an interface for all benchmarks,
    to allow them to be called easily and in a structured way.
    """

    @abc.abstractmethod
    def is_trial_appropriate(self, trial_result: arvet.core.trial_result.TrialResult) -> bool:
        """
        Fine-grained filtering for trial results, to make sure this class can benchmark this trial result.
        :return: 
        """
        pass

    @abc.abstractmethod
    def benchmark_results(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) \
            -> 'BenchmarkResult':
        """
        Benchmark the results of running a particular system on a particular image source.
        We take a collection of trials to allow for multiple repeats of the system on the same data,
        which allows us to account for and measure random variation in the system.
        A helper to check this is provided below, call it in any implementation.

        The trial result MUST include the ground truth along with the system estimates,
        which must be the same for all trials.

        :param trial_results: A collection of trial results to benchmark.
        These are assumed to be repeat runs of the same system on the same data.
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass


class BenchmarkResult(arvet.database.entity.Entity):
    """
    A general superclass for benchmark results for all benchmarks
    """
    def __init__(self, benchmark_id: bson.ObjectId, trial_result_ids: typing.Iterable[bson.ObjectId], success: bool,
                 id_: bson.ObjectId = None, **kwargs):
        """

        :param benchmark_id: The identifier for the benchmark producing this result
        :param trial_result_ids: The ids of the trial results measured by this benchmark result
        :param success: Did the benchmark succeed. Everything not a subtype of FailedBenchmark should pass true.
        """
        super().__init__(id_=id_, **kwargs)
        self._success = success
        self._benchmark = benchmark_id
        self._trial_results = set(trial_result_ids)

    @property
    def benchmark(self) -> bson.ObjectId:
        return self._benchmark

    @property
    def trial_results(self) -> typing.Set[bson.ObjectId]:
        return self._trial_results

    @property
    def success(self) -> bool:
        return self._success

    def serialize(self):
        serialized = super().serialize()
        serialized['success'] = self._success
        serialized['benchmark'] = self._benchmark
        serialized['trial_results'] = list(self._trial_results)
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'success' in serialized_representation:
            kwargs['success'] = serialized_representation['success']
        if 'benchmark' in serialized_representation:
            kwargs['benchmark_id'] = serialized_representation['benchmark']
        if 'trial_results' in serialized_representation:
            kwargs['trial_result_ids'] = serialized_representation['trial_results']
        return super().deserialize(serialized_representation, db_client, **kwargs)


class FailedBenchmark(BenchmarkResult):
    """
    A superclass for all benchmark results that represent failure in some way.

    Think of this a bit like an exception, all BenchmarkResults should have success = false iff they
    inherit from this class.
    """

    def __init__(self, benchmark_id: bson.ObjectId, trial_result_ids: typing.Iterable[bson.ObjectId], reason: str,
                 id_: bson.ObjectId = None, **kwargs):
        """

        :param reason: String explaining what went wrong.
        """
        kwargs['success'] = False
        super().__init__(benchmark_id, trial_result_ids, id_=id_, **kwargs)
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    def serialize(self):
        serialized = super().serialize()
        serialized['reason'] = self.reason
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'reason' in serialized_representation:
            kwargs['reason'] = serialized_representation['reason']
        return super().deserialize(serialized_representation, db_client, **kwargs)


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
