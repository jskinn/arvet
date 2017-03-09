from unittest import TestCase
from core.benchmark import BenchmarkResult, FailedBenchmark


class TestBenchmarkResult(TestCase):

    def test_no_id(self):
        BenchmarkResult(1, 2, True)

    def test_identifier(self):
        benchmark_result = BenchmarkResult(1, 2, True, id_=123)
        self.assertEquals(benchmark_result.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a':1, 'b':2, 'c': 3}
        with self.assertRaises(TypeError):
            BenchmarkResult(1, 2, True, **kwargs)

    def test_serialize_and_deserialize(self):
        benchmark_result1 = BenchmarkResult(1, 2, True, id_=12345)
        s_benchmark_result1 = benchmark_result1.serialize()

        benchmark_result2 = BenchmarkResult.deserialize(s_benchmark_result1)
        s_benchmark_result2 = benchmark_result2.serialize()

        self._assert_models_equal(benchmark_result1, benchmark_result2)
        self.assertEquals(s_benchmark_result1, s_benchmark_result2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            benchmark_result2 = BenchmarkResult.deserialize(s_benchmark_result2)
            s_benchmark_result2 = benchmark_result2.serialize()
            self._assert_models_equal(benchmark_result1, benchmark_result2)
            self.assertEquals(s_benchmark_result1, s_benchmark_result2)

    def _assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if not isinstance(benchmark_result1, BenchmarkResult) or not isinstance(benchmark_result2, BenchmarkResult):
            self.fail('object was not a BenchmarkResult')
        self.assertEquals(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEquals(benchmark_result1.success, benchmark_result2.success)
        self.assertEquals(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEquals(benchmark_result1.trial_result, benchmark_result2.trial_result)

class TestFailed(TestCase):

    def test_no_id(self):
        FailedBenchmark(1, 2, 'For testing')

    def test_success_false(self):
        benchmark_result = FailedBenchmark(1, 2, 'For testing')
        self.assertFalse(benchmark_result.success)

    def test_identifier(self):
        benchmark_result = FailedBenchmark(1, 2, 'For testing', id_=123)
        self.assertEquals(benchmark_result.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a':1, 'b':2, 'c': 3}
        benchmark_result = FailedBenchmark(1, 2, 'For testing', **kwargs)
        self.assertEquals(benchmark_result.identifier, 1234)

    def test_serialize_and_deserialize(self):
        benchmark_result1 = FailedBenchmark(1, 2, 'For testing', id_=12345)
        s_benchmark_result1 = benchmark_result1.serialize()

        benchmark_result2 = FailedBenchmark.deserialize(s_benchmark_result1)
        s_benchmark_result2 = benchmark_result2.serialize()

        self.assertFalse(benchmark_result2.success)
        self._assert_models_equal(benchmark_result1, benchmark_result2)
        self.assertEquals(s_benchmark_result1, s_benchmark_result2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            benchmark_result2 = FailedBenchmark.deserialize(s_benchmark_result2)
            s_benchmark_result2 = benchmark_result2.serialize()
            self._assert_models_equal(benchmark_result1, benchmark_result2)
            self.assertEquals(s_benchmark_result1, s_benchmark_result2)

    def _assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two dataset models are equal
        :param benchmark_result1: Dataset
        :param benchmark_result2: Dataset
        :return:
        """
        if not isinstance(benchmark_result1, FailedBenchmark) or not isinstance(benchmark_result2, FailedBenchmark):
            self.fail('object was not a FailedBenchmark')
        self.assertEquals(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEquals(benchmark_result1.success, benchmark_result2.success)
        self.assertEquals(benchmark_result1.reason, benchmark_result2.reason)
        self.assertEquals(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEquals(benchmark_result1.trial_result, benchmark_result2.trial_result)