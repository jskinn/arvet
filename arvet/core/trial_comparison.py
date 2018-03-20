# Copyright (c) 2017, John Skinner
import abc
import bson
import typing
import arvet.database.entity
import arvet.core.trial_result
import arvet.core.benchmark


class TrialComparison(arvet.database.entity.Entity, metaclass=abc.ABCMeta):
    """
    Some benchmarks and performance measures only make sense when comparing two different sets of trial results,
    that is, by comparing two similar runs and measuring the difference in some way.
    Benchmarks of this type take two different trial results, and measures the difference between them.

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
    def compare_trial_results(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult],
                              reference_trial_result: typing.Iterable[arvet.core.trial_result.TrialResult]) -> \
            arvet.core.benchmark.BenchmarkResult:
        """
        Compare the results of the first group of trials with a group of reference trials.
        Should return a FailedBenchmark if there is a problem.

        :param trial_results: An iterable of TrialResults to compare
        :param reference_trial_result: An iterable of TrialResult to compare to
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass


class TrialComparisonResult(arvet.core.benchmark.BenchmarkResult):
    """
    A general superclass for benchmark results that compare two groups trials.
    """
    def __init__(self, benchmark_id: bson.ObjectId, trial_result_ids: typing.Iterable[bson.ObjectId],
                 reference_ids: typing.Iterable[bson.ObjectId], success: bool, id_: bson.ObjectId = None, **kwargs):
        """
        :param benchmark_id: The TrialComparison benchmark that produced this result
        :param trial_result_id: The first TrialResult, which is compared to the reference
        :param reference_id: The reference TrialResult, to which the first is compared
        :param success: Did the benchmark succeed. Everything not a subtype of FailedBenchmark should pass true.
        :param id_: The ID of the TrialComparisonResult, if it exists.
        """
        super().__init__(benchmark_id, trial_result_ids, success, id_, **kwargs)
        self._reference_ids = set(reference_ids)

    @property
    def reference_trial_results(self) -> typing.Set[bson.ObjectId]:
        """
        The id of the reference trial to which the second trial is compared.
        This affects the order of the measured difference
        :return:
        """
        return self._reference_ids

    def save_data(self, db_client):
        """
        Like trial results, benchmark results can be really large,
        provide an opportunity to save the data elsewhere.
        :param db_client: The database client.
        :return:
        """
        pass

    def serialize(self):
        serialized = super().serialize()
        serialized['reference'] = self._reference_ids
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'reference' in serialized_representation:
            kwargs['reference_ids'] = serialized_representation['reference']
        return super().deserialize(serialized_representation, db_client, **kwargs)
