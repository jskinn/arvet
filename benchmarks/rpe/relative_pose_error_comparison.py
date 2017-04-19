import numpy as np
import pickle
import bson
import core.benchmark_comparison


class RPEBenchmarkComparison(core.benchmark_comparison.BenchmarkComparison):
    """
    Comparison of two Relative Pose Error benchmark results.
    Basically just the difference in translational error.
    """

    @property
    def identifier(self):
        return 'RelativePoseErrorComparison'

    def get_trial_requirements(self):
        """
        Get the requirements for benchmark results that can be compared by this BenchmarkComparison.
        Both benchmark results must be RelativePoseError results.
        :return: 
        """
        return {'benchmark': 'RelativePoseError'}

    def compare_trial_results(self, benchmark_result, reference_benchmark_result):
        """
        Compare the first Relative Pose Error result with a reference benchmark result.
        :param benchmark_result: 
        :param reference_benchmark_result: 
        :return: 
        """

        #TODO: Need to match the timestamps between the test and reference results.

        trans_error_diff = reference_benchmark_result.translational_error - benchmark_result.translational_error
        rot_error_diff = reference_benchmark_result.rotational_error - benchmark_result.rotational_error
        return RPEBenchmarkComparisonResult(benchmark_comparison_id=self.identifier,
                                            benchmark_result=benchmark_result.identifier,
                                            reference_benchmark_result=reference_benchmark_result.identifier,
                                            difference_in_translational_error=trans_error_diff,
                                            difference_in_rotational_error=rot_error_diff)


class RPEBenchmarkComparisonResult(core.benchmark_comparison.BenchmarkComparisonResult):
    """
    The result of comparing two Relative Pose Error measurements.
    Is just the difference in each of the error metrics. 
    """

    def __init__(self, benchmark_comparison_id, benchmark_result, reference_benchmark_result,
                 difference_in_translational_error, difference_in_rotational_error, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_comparison_id=benchmark_comparison_id,
                         benchmark_result=benchmark_result,
                         reference_benchmark_result=reference_benchmark_result,
                         id_=id_, **kwargs)
        self._trans_error_diff = difference_in_translational_error
        self._rot_error_diff = difference_in_rotational_error

    @property
    def translational_error_difference(self):
        return self._trans_error_diff

    @property
    def trans_rmse(self):
        return np.sqrt(np.dot(self.translational_error_difference, self.translational_error_difference) /
                       len(self.translational_error_difference))

    @property
    def trans_mean(self):
        return np.mean(self.translational_error_difference)

    @property
    def trans_median(self):
        return np.median(self.translational_error_difference)

    @property
    def trans_std(self):
        return np.std(self.translational_error_difference)

    @property
    def trans_min(self):
        return np.min(self.translational_error_difference)

    @property
    def trans_max(self):
        return np.max(self.translational_error_difference)

    @property
    def rotational_error_difference(self):
        return self._rot_error_diff

    @property
    def rot_rmse(self):
        return np.sqrt(np.dot(self.rotational_error_difference, self.rotational_error_difference) /
                       len(self.rotational_error_difference))

    @property
    def rot_mean(self):
        return np.mean(self.rotational_error_difference)

    @property
    def rot_median(self):
        return np.median(self.rotational_error_difference)

    @property
    def rot_std(self):
        return np.std(self.rotational_error_difference)

    @property
    def rot_min(self):
        return np.min(self.rotational_error_difference)

    @property
    def rot_max(self):
        return np.max(self.rotational_error_difference)

    def serialize(self):
        output = super().serialize()
        output['trans_error_diff'] = bson.Binary(pickle.dumps(self.translational_error_difference,
                                                              protocol=pickle.HIGHEST_PROTOCOL))
        output['rot_error_diff'] = bson.Binary(pickle.dumps(self.rotational_error_difference,
                                                            protocol=pickle.HIGHEST_PROTOCOL))
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'trans_error_diff' in serialized_representation:
            kwargs['difference_in_translational_error'] = pickle.loads(serialized_representation['trans_error_diff'])
        if 'rot_error_diff' in serialized_representation:
            kwargs['difference_in_rotational_error'] = pickle.loads(serialized_representation['rot_error_diff'])
        return super().deserialize(serialized_representation, **kwargs)
