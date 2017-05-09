import numpy as np
import pickle
import bson
import util.associate as ass
import core.benchmark_comparison


class RPEBenchmarkComparison(core.benchmark_comparison.BenchmarkComparison):
    """
    Comparison of two Relative Pose Error benchmark results.
    Basically just the difference in translational error.
    """

    def __init__(self, offset=0, max_difference=0.02, id_=None):
        """
        Make a Comparison Benchmark for RPE,
        parameters are for configuring the matches between the two compared benchmarks 
        :param offset: offset applied to the computed benchmark timestamps
        :param max_difference: Maximum acceptable difference between timestamps
        """
        super().__init__(id_=id_)
        self._offset = offset
        self._max_difference = max_difference

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def max_difference(self):
        return self._max_difference

    @max_difference.setter
    def max_difference(self, max_difference):
        if max_difference >= 0:
            self._max_difference = max_difference

    def get_settings(self):
        return {
            'offset': self.offset,
            'max_difference': self.max_difference
        }

    def serialize(self):
        output = super().serialize()
        output['offset'] = self.offset
        output['max_difference'] = self.max_difference
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'offset' in serialized_representation:
            kwargs['offset'] = serialized_representation['offset']
        if 'max_difference' in serialized_representation:
            kwargs['max_difference'] = serialized_representation['max_difference']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_benchmark_requirements(cls):
        """
        Get the requirements for benchmark results that can be compared by this BenchmarkComparison.
        Both benchmark results must be RelativePoseError results.
        :return: 
        """
        return {'benchmark': 'RelativePoseError'}

    def is_result_appropriate(self, benchmark_result):
        """
        Can this particular benchmark result be used in the benchmark?
        :param benchmark_result: 
        :return: 
        """
        return (hasattr(benchmark_result, 'identifier') and
                hasattr(benchmark_result, 'translational_error') and
                hasattr(benchmark_result, 'rotational_error'))

    def compare_results(self, benchmark_result, reference_benchmark_result):
        """
        Compare the first Relative Pose Error result with a reference benchmark result.
        :param benchmark_result: 
        :param reference_benchmark_result: 
        :return: 
        """
        matches = ass.associate(benchmark_result.translational_error, reference_benchmark_result.translational_error,
                                offset=self.offset, max_difference=self.max_difference)
        trans_error_diff = {}
        rot_error_diff = {}
        for result_stamp, ref_stamp in matches:
            trans_error_diff[ref_stamp] = (reference_benchmark_result.translational_error[ref_stamp] -
                                           benchmark_result.translational_error[result_stamp])
            rot_error_diff[ref_stamp] = (reference_benchmark_result.rotational_error[ref_stamp] -
                                         benchmark_result.rotational_error[result_stamp])
        return RPEBenchmarkComparisonResult(benchmark_comparison_id=self.identifier,
                                            benchmark_result=benchmark_result.identifier,
                                            reference_benchmark_result=reference_benchmark_result.identifier,
                                            difference_in_translational_error=trans_error_diff,
                                            difference_in_rotational_error=rot_error_diff,
                                            settings=self.get_settings())


class RPEBenchmarkComparisonResult(core.benchmark_comparison.BenchmarkComparisonResult):
    """
    The result of comparing two Relative Pose Error measurements.
    Is just the difference in each of the error metrics. 
    """

    def __init__(self, benchmark_comparison_id, benchmark_result, reference_benchmark_result,
                 difference_in_translational_error, difference_in_rotational_error, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_comparison_id=benchmark_comparison_id,
                         benchmark_result=benchmark_result,
                         reference_benchmark_result=reference_benchmark_result,
                         id_=id_, **kwargs)
        self._trans_error_diff = difference_in_translational_error
        self._rot_error_diff = difference_in_rotational_error
        self._settings = settings

    @property
    def translational_error_difference(self):
        return self._trans_error_diff

    @property
    def trans_rmse(self):
        trans_error = np.array(list(self.translational_error_difference.values()))
        return np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))

    @property
    def trans_mean(self):
        trans_error = np.array(list(self.translational_error_difference.values()))
        return np.mean(trans_error)

    @property
    def trans_median(self):
        trans_error = np.array(list(self.translational_error_difference.values()))
        return np.median(trans_error)

    @property
    def trans_std(self):
        trans_error = np.array(list(self.translational_error_difference.values()))
        return np.std(trans_error)

    @property
    def trans_min(self):
        trans_error = np.array(list(self.translational_error_difference.values()))
        return np.min(trans_error)

    @property
    def trans_max(self):
        trans_error = np.array(list(self.translational_error_difference.values()))
        return np.max(trans_error)

    @property
    def rotational_error_difference(self):
        return self._rot_error_diff

    @property
    def rot_rmse(self):
        rot_error = np.array(list(self.rotational_error_difference.values()))
        return np.sqrt(np.dot(rot_error, rot_error) / len(rot_error))

    @property
    def rot_mean(self):
        rot_error = np.array(list(self.rotational_error_difference.values()))
        return np.mean(rot_error)

    @property
    def rot_median(self):
        rot_error = np.array(list(self.rotational_error_difference.values()))
        return np.median(rot_error)

    @property
    def rot_std(self):
        rot_error = np.array(list(self.rotational_error_difference.values()))
        return np.std(rot_error)

    @property
    def rot_min(self):
        rot_error = np.array(list(self.rotational_error_difference.values()))
        return np.min(rot_error)

    @property
    def rot_max(self):
        rot_error = np.array(list(self.rotational_error_difference.values()))
        return np.max(rot_error)

    @property
    def settings(self):
        return self._settings

    def serialize(self):
        output = super().serialize()
        output['trans_error_diff'] = bson.Binary(pickle.dumps(self.translational_error_difference,
                                                              protocol=pickle.HIGHEST_PROTOCOL))
        output['rot_error_diff'] = bson.Binary(pickle.dumps(self.rotational_error_difference,
                                                            protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trans_error_diff' in serialized_representation:
            kwargs['difference_in_translational_error'] = pickle.loads(serialized_representation['trans_error_diff'])
        if 'rot_error_diff' in serialized_representation:
            kwargs['difference_in_rotational_error'] = pickle.loads(serialized_representation['rot_error_diff'])
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
