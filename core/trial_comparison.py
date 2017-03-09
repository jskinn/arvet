import abc
import database.identifiable
import core.benchmark


class TrialComparison(database.identifiable.Identifiable, metaclass=abc.ABCMeta):
    """
    Some benchmarks and performance measures only make sense when comparing two different trial results,
    that is, by comparing two similar runs and measuring the difference in some way.
    Benchmarks of this type take two different trial results, and measures the difference between them.

    This is an abstract base class defining an interface for all such benchmarks,
    to allow them to be called easily and in a structured way.
    """

    @abc.abstractmethod
    def get_trial_requirements(self):
        """
        Get the requirements to determine which trial_results are relevant for this benchmark.
        Should return a dict that is a mongodb query operator
        :return:
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def compare_trial_results(self, trial_result, reference_trial_result):
        """
        Compare the results of the first trial with a reference trial.
        Should return a FailedBenchmark if there is a problem.

        :param trial_result: TrialResult
        :param reference_trial_result: TrialResult
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass


class TrialComparisonResult(core.benchmark.BenchmarkResult):
    """
    A general superclass for benchmark results that compare two trials.
    """
    def __init__(self, benchmark_id, trial_result_id, reference_id, success, id_=None, **kwargs):
        """
        :param benchmark_id: The TrialComparison benchmark that produced this result
        :param trial_result_id: The first TrialResult, which is compared to the reference
        :param reference_id: The reference TrialResult, to which the first is compared
        :param success: Did the benchmark succeed. Everything not a subtype of FailedBenchmark should pass true.
        :param id_: The ID of the TrialComparisonResult, if it exists.
        """
        super().__init__(benchmark_id, trial_result_id, success, id_, **kwargs)
        self._reference_id = reference_id

    @property
    def reference_trial_result(self):
        """
        The id of the reference trial to which the second trial is compared.
        This affects the order of the measured difference
        :return:
        """
        return self._reference_id

    def serialize(self):
        serialized = super().serialize()
        serialized['reference'] = self._reference_id
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'reference' in serialized_representation:
            kwargs['reference_id'] = serialized_representation['reference']
        return super().deserialize(serialized_representation, **kwargs)
