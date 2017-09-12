#Copyright (c) 2017, John Skinner
import numpy as np
import pickle
import bson
import util.associate as ass
import core.benchmark_comparison


class ATEBenchmarkComparison(core.benchmark_comparison.BenchmarkComparison):
    """
    Comparison of two Absolute Trajectory Error benchmark results.
    Basically just the difference in translational error.
    """

    def __init__(self, offset=0, max_difference=0.02, id_=None):
        """
        Make a Comparison Benchmark for ATE,
        parameters are for configuring the matches between the two compared benchmarks 
        :param offset: Offset between
        :param max_difference:
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
        Both benchmark results must be AbsoluteTrajectoryError results.
        :return: 
        """
        return {'benchmark': 'AbsoluteTrajectoryError'}

    def is_result_appropriate(self, benchmark_result):
        """
        Can this particular benchmark result be used in the benchmark?
        :param benchmark_result: 
        :return: 
        """
        return hasattr(benchmark_result, 'identifier') and hasattr(benchmark_result, 'translational_error')

    def compare_results(self, benchmark_result, reference_benchmark_result):
        """
        Compare the first Absolute Trajectory Error result with a reference benchmark result.
        :param benchmark_result: 
        :param reference_benchmark_result: 
        :return: 
        """
        matches = ass.associate(reference_benchmark_result.translational_error, benchmark_result.translational_error,
                                offset=self.offset, max_difference=self.max_difference)

        trans_error_diff = {}
        for ref_stamp, result_stamp in matches:
            trans_error_diff[ref_stamp] = (reference_benchmark_result.translational_error[ref_stamp] -
                                           benchmark_result.translational_error[result_stamp])
        return ATEBenchmarkComparisonResult(benchmark_comparison_id=self.identifier,
                                            benchmark_result=benchmark_result.identifier,
                                            reference_benchmark_result=reference_benchmark_result.identifier,
                                            difference_in_translational_error=trans_error_diff,
                                            settings=self.get_settings())


class ATEBenchmarkComparisonResult(core.benchmark_comparison.BenchmarkComparisonResult):
    """
    The result of comparing two Absolute Trajectory Error measurements.
    Is just the difference in translational error 
    """

    def __init__(self, benchmark_comparison_id, benchmark_result, reference_benchmark_result,
                 difference_in_translational_error, settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(benchmark_comparison_id=benchmark_comparison_id,
                         benchmark_result=benchmark_result,
                         reference_benchmark_result=reference_benchmark_result,
                         id_=id_, **kwargs)
        self._trans_error_diff = difference_in_translational_error
        self._settings = settings

        # Compute and cache error statistics
        error_array = np.array(list(difference_in_translational_error.values()))
        self._rmse = np.sqrt(np.dot(error_array, error_array) / len(difference_in_translational_error))
        self._mean = np.mean(error_array)
        self._median = np.median(error_array)
        self._std = np.std(error_array)
        self._min = np.min(error_array)
        self._max = np.max(error_array)

    @property
    def translational_error_difference(self):
        return self._trans_error_diff

    @property
    def settings(self):
        return self._settings

    @property
    def num_pairs(self):
        return len(self.translational_error_difference)

    @property
    def rmse(self):
        return self._rmse

    @property
    def mean(self):
        return self._mean

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
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trans_error_diff' in serialized_representation:
            kwargs['difference_in_translational_error'] = pickle.loads(serialized_representation['trans_error_diff'])
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
