from abc import ABCMeta, abstractmethod
from database.entity import Entity
from database.referenced_class import ReferencedClass


class Benchmark(ReferencedClass, metaclass=ABCMeta):
    """
    A class that benchmarks SLAM algorithms.

    This is an abstract base class defining an interface for all benchmarks,
    to allow them to be called easily and in a structured way.
    """

    @abstractmethod
    def get_trial_requirements(self):
        """
        Get the requirements to determine which trial_results are relevant for this benchmark.
        Should return a dict that is a mongodb query operator
        :return:
        :rtype: dict
        """
        pass

    @abstractmethod
    def benchmark_results(self, dataset_images, trial_result):
        """
        Benchmark the result of a particular trial

        :param dataset_images: DatasetImageSet The dataset with loaded images that produced the trial result
        :param trial_result: TrialResult
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass


class BenchmarkResult(Entity):
    """
    A general superclass for benchmark results for all
    """
    def __init__(self, benchmark_id, trial_result_id, success, id_=None, **kwargs):
        """

        :param success: Did the benchmark succeeed. Everything not a subtype of FailedBenchmark should pass true.
        """
        super().__init__(id_)
        self._success = success
        self._benchmark = benchmark_id
        self._trial_result = trial_result_id

    @property
    def benchmark(self):
        return self._benchmark

    @property
    def trial_result(self):
        return self._trial_result

    @property
    def success(self):
        return self._success

    def serialize(self):
        serialized = super().serialize()
        serialized['success'] = self._success
        serialized['benchmark'] = self._benchmark
        serialized['trial_result'] = self._trial_result
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'success' in serialized_representation:
            kwargs['success'] = serialized_representation['success']
        if 'benchmark' in serialized_representation:
            kwargs['benchmark_id'] = serialized_representation['benchmark']
        if 'trial_result' in serialized_representation:
            kwargs['trial_result_id'] = serialized_representation['trial_result']
        return super().deserialize(serialized_representation, **kwargs)


class FailedBenchmark(BenchmarkResult):
    """
    A superclass for all benchmark results that represent failure in some way.

    Think of this a bit like an exception, all BenchmarkResults should have success = false iff they
    inherit from this class.
    """

    def __init__(self, benchmark_id, trial_result_id, reason, id_=None, **kwargs):
        """

        :param reason:
        :return: FailedBenchmark
        :rtype: FailedBenchmark
        """
        super().__init__(benchmark_id, trial_result_id, False, id_)
        self._reason = reason

    @property
    def reason(self):
        return self._reason

    def serialize(self):
        serialized = super().serialize()
        serialized['reason'] = self.reason
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'reason' in serialized_representation:
            kwargs['reason'] = serialized_representation['reason']
        return super().deserialize(serialized_representation, **kwargs)
