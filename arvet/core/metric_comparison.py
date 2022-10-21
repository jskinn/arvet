# Copyright (c) 2017, John Skinner
import abc
import pymodm
import pymodm.fields as fields
import typing
import arvet.database.pymodm_abc as pymodm_abc
import arvet.core.trial_result
from arvet.core.metric import MetricResult


class MetricComparison(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    Some metrics, rather than reading the output of the system directly, instead compare the results of  other metrics

    These are similar to TrialComparisonMetric, but save re-implementing a base benchmark,
    and let me just subtract one from another.

    This is an abstract base class defining an interface for all such benchmarks,
    to allow them to be called easily and in a structured way.
    """

    @abc.abstractmethod
    def is_result_appropriate(self, metric_result: MetricResult) -> bool:
        """
        Fine-grained filtering of which metric results this comparison can be applied to.

        :param metric_result:
        :return: 
        """
        pass

    @abc.abstractmethod
    def compare_results(self, metric_results_1: typing.Iterable[MetricResult],
                        metric_results_2: typing.Iterable[MetricResult]
                        ) -> 'MetricComparisonResult':
        """
        Compare the benchmark of one trial with the benchmark of another trial.
        Should return a FailedBenchmark if there is a problem.

        :param benchmark_result: The benchmark result to compare
        :param reference_benchmark_result: The reference benchmark result to compare to
        :return: A BenchmarkResult object containing either the results, or a FailedBenchmark explaining the error
        :rtype: BenchmarkResult
        """
        pass


class MetricComparisonResult(pymodm.MongoModel):
    """
    Results for metrics that compare other metrics
    """
    metric = fields.ReferenceField(MetricComparison, required=True, on_delete=fields.ReferenceField.CASCADE)
    Metri = fields.ListField(fields.ReferenceField(arvet.core.trial_result.TrialResult,
                                                           required=True, on_delete=fields.ReferenceField.CASCADE))
    success = fields.BooleanField(required=True)
    message = fields.CharField()
