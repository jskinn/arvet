import numpy as np
import pickle
import bson
import core.benchmark_comparison


class ATEBenchmarkComparison(core.benchmark_comparison.BenchmarkComparison):
    """
    Comparison of two Absolute Trajectory Error benchmark results.
    Basically just the difference in translational error.
    """

    @property
    def identifier(self):
        return 'AbsoluteTrajectoryErrorComparison'

    def get_trial_requirements(self):
        """
        Get the requirements for benchmark results that can be compared by this BenchmarkComparison.
        Both benchmark results must be AbsoluteTrajectoryError results.
        :return: 
        """
        return {'benchmark': 'AbsoluteTrajectoryError'}

    def compare_trial_results(self, benchmark_result, reference_benchmark_result):
        """
        Compare the first Absolute Trajectory Error result with a reference benchmark result.
        :param benchmark_result: 
        :param reference_benchmark_result: 
        :return: 
        """
        trans_error_diff = reference_benchmark_result.translational_error - benchmark_result.translational_error
        return BenchmarkATEComparisonResult(benchmark_comparison_id=self.identifier,
                                            benchmark_result=benchmark_result.identifier,
                                            reference_benchmark_result=reference_benchmark_result.identifier,
                                            difference_in_translational_error=trans_error_diff)


class BenchmarkATEComparisonResult(core.benchmark_comparison.BenchmarkComparisonResult):
    """
    The result of comparing two Absolute Trajectory Error measurements.
    Is just the difference in translational error 
    """

    def __init__(self, benchmark_comparison_id, benchmark_result, reference_benchmark_result,
                 difference_in_translational_error, id_=None):
        super().__init__(benchmark_comparison_id, benchmark_result, reference_benchmark_result, True, id_)
        self._trans_error_diff = difference_in_translational_error

    @property
    def translational_error_difference(self):
        return self._trans_error_diff

    @property
    def num_pairs(self):
        return len(self.translational_error_difference)

    @property
    def rmse(self):
        return np.sqrt(np.dot(self.translational_error_difference, self.translational_error_difference) /
                       self.num_pairs)

    @property
    def mean(self):
        return np.mean(self.translational_error_difference)

    @property
    def median(self):
        return np.median(self.translational_error_difference)

    @property
    def std(self):
        return np.std(self.translational_error_difference)

    @property
    def min(self):
        return np.min(self.translational_error_difference)

    @property
    def max(self):
        return np.max(self.translational_error_difference)

    def serialize(self):
        output = super().serialize()
        output['trans_error_diff'] = bson.Binary(pickle.dumps(self.translational_error_difference,
                                                              protocol=pickle.HIGHEST_PROTOCOL))
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'trans_error_diff' in serialized_representation:
            kwargs['difference_in_translational_error'] = pickle.loads(serialized_representation['trans_error_diff'])
        return super().deserialize(serialized_representation, **kwargs)
