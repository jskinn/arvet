# Copyright (c) 2017, John Skinner
import abc
import bson
import arvet.database.entity
import arvet.core.benchmark


class BenchmarkComparison(arvet.database.entity.Entity, metaclass=abc.ABCMeta):
    """
    Some benchmarks and performance measures only make sense when comparing two different trial results,
    that is, by comparing two similar runs and measuring the difference in some way.
    Benchmarks of this type take two different benchmark results, and measures the difference between them.

    These are similar to TrialComparisons, but save re-implementing a base benchmark,
    and let me just subtract one from another.

    This is an abstract base class defining an interface for all such benchmarks,
    to allow them to be called easily and in a structured way.
    """

    @abc.abstractmethod
    def is_result_appropriate(self, benchmark_result: arvet.core.benchmark.BenchmarkResult) -> bool:
        """
        Fine-grained filtering of which benchmarks results this comparison can be applied to.
        Has access to the deserialized benchmark result, with all it's properties
        :param benchmark_result: 
        :return: 
        """
        pass

    @abc.abstractmethod
    def compare_results(self, benchmark_result: arvet.core.benchmark.BenchmarkResult,
                        reference_benchmark_result: arvet.core.benchmark.BenchmarkResult
                        ) -> 'BenchmarkComparisonResult':
        """
        Compare the benchmark of one trial with the benchmark of another trial.
        Should return a FailedBenchmark if there is a problem.

        :param benchmark_result: The benchmark result to compare
        :param reference_benchmark_result: The reference benchmark result to compare to
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass


class BenchmarkComparisonResult(arvet.database.entity.Entity):
    """
    A general superclass for benchmark results for all
    """
    def __init__(self, benchmark_comparison_id: bson.ObjectId, benchmark_result: bson.ObjectId,
                 reference_benchmark_result: bson.ObjectId, success: bool, id_: bson.ObjectId = None, **kwargs):
        """

        :param success: Did the benchmark succeed. Everything not a subtype of FailedBenchmark should pass true.
        """
        super().__init__(id_=id_, **kwargs)
        self._benchmark_comparison_id = benchmark_comparison_id
        self._benchmark_id = benchmark_result
        self._reference_id = reference_benchmark_result
        self._success = success

    @property
    def comparison_id(self) -> bson.ObjectId:
        """
        The id of the benchmark comparison used to compare the two benchmark results
        :return: ObjectID or string
        """
        return self._benchmark_comparison_id

    @property
    def benchmark_result(self) -> bson.ObjectId:
        """
        The id of the query benchmark that is compared to the reference benchmark.
        :return: ObjectId
        """
        return self._benchmark_id

    @property
    def reference_benchmark_result(self) -> bson.ObjectId:
        """
        The id of the reference benchmark to which the first benchmark is compared.
        This affects the order of the measured difference
        :return: ObjectId
        """
        return self._reference_id

    @property
    def success(self) -> bool:
        return self._success

    def serialize(self):
        serialized = super().serialize()
        serialized['benchmark_comparison'] = self.comparison_id
        serialized['benchmark_result'] = self.benchmark_result
        serialized['reference_benchmark_result'] = self.reference_benchmark_result
        serialized['success'] = self.success
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'benchmark_result' in serialized_representation:
            kwargs['benchmark_result'] = serialized_representation['benchmark_result']
        if 'reference_benchmark_result' in serialized_representation:
            kwargs['reference_benchmark_result'] = serialized_representation['reference_benchmark_result']
        if 'benchmark_comparison' in serialized_representation:
            kwargs['benchmark_comparison_id'] = serialized_representation['benchmark_comparison']
        if 'success' in serialized_representation:
            kwargs['success'] = serialized_representation['success']
        return super().deserialize(serialized_representation, db_client, **kwargs)
