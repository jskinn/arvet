import abc
import database.referenced_class
import core.benchmark


class ComparisonBenchmark(database.referenced_class.ReferencedClass, metaclass=abc.ABCMeta):
    """
    Some benchmarks and performance measures only make sense when comparing two different results,
    that is, by comparing two similar runs and measuring the difference in some way.

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
    def compare_results(self, trial_result, reference_trial_result, reference_dataset_images):
        """
        Benchmark the result of a particular trial

        :param trial_result: TrialResult
        :param reference_trial_result: TrialResult
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass

class ComparisonBenchmarkResult(core.benchmark.BenchmarkResult):
    """
    A general superclass for benchmark results for all
    """
    def __init__(self, benchmark_id, trial_result_id, reference_id, success, id_=None, **kwargs):
        """

        :param success: Did the benchmark succeed. Everything not a subtype of FailedBenchmark should pass true.
        """
        super().__init__(benchmark_id, trial_result_id, success, id_)
        self._reference_id = reference_id

    @property
    def reference_trial_result(self):
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
