import batch_analysis.task


class CompareBenchmarksTask(batch_analysis.task.Task):
    """
    A task for comparing two benchmark results against each other. Result is a BenchmarkComparison id
    """
    def __init__(self, benchmark_result1_id, benchmark_result2_id, comparison_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._benchmark_result1_id = benchmark_result1_id
        self._benchmark_result2_id = benchmark_result2_id
        self._comparison_id = comparison_id

    @property
    def benchmark_result1(self):
        return self._benchmark_result1_id

    @property
    def benchmark_result2(self):
        return self._benchmark_result2_id

    @property
    def comparison(self):
        return self._comparison_id

    def run_task(self, db_client):
        import logging
        import traceback
        import util.database_helpers as dh

        benchmark_result_1 = dh.load_object(db_client, db_client.results_collection, self.benchmark_result1)
        benchmark_result_2 = dh.load_object(db_client, db_client.results_collection, self.benchmark_result2)
        comparison_benchmark = dh.load_object(db_client, db_client.benchmarks_collection, self.comparison)

        if benchmark_result_1 is None:
            logging.getLogger(__name__).error("Could not deserialize benchmark result {0}".format(
                self.benchmark_result1))
            self.mark_job_failed()
        elif benchmark_result_2 is None:
            logging.getLogger(__name__).error("Could not deserialize benchmark result {0}".format(
                self.benchmark_result2))
            self.mark_job_failed()
        elif comparison_benchmark is None:
            logging.getLogger(__name__).error("Could not deserialize comparison benchmark {0}".format(self.comparison))
            self.mark_job_failed()
        elif (not comparison_benchmark.is_trial_appropriate(benchmark_result_1) or
                not comparison_benchmark.is_trial_appropriate(benchmark_result_2)):
            logging.getLogger(__name__).error("Benchmark {0} is not appropriate for"
                                              "benchmark results {1} or {2}".format(self.comparison,
                                                                                    self.benchmark_result1,
                                                                                    self.benchmark_result2))
            self.mark_job_failed()
        else:
            logging.getLogger(__name__).info("Comparing benchmark results {0} and {1}"
                                             "with comparison benchmark {2}".format(self.benchmark_result1,
                                                                                    self.benchmark_result2,
                                                                                    self.comparison))
            try:
                comparison_result = comparison_benchmark.compare_results(benchmark_result_1, benchmark_result_2)
            except Exception:
                logging.getLogger(__name__).error("Error occurred while comparing benchmark results {0} and {1}"
                                                  "with benchmark {2}:\n{3}".format(self.benchmark_result1,
                                                                                    self.benchmark_result2,
                                                                                    self.comparison,
                                                                                    traceback.format_exc()))
                comparison_result = None
            if comparison_result is None:
                logging.getLogger(__name__).error("Failed to compare benchmarks {0} and {1} with {2}".format(
                    self.benchmark_result1, self.benchmark_result2, self.comparison))
                self.mark_job_failed()
            else:
                result_id = db_client.results_collection.insert(comparison_result.serialize())
                logging.getLogger(__name__).info("Successfully compared benchmark results {0} and {1},"
                                                 "producing result {2}".format(self.benchmark_result1,
                                                                               self.benchmark_result2,
                                                                               result_id))
                self.mark_job_complete(result_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['benchmark_result1_id'] = self.benchmark_result1
        serialized['benchmark_result2_id'] = self.benchmark_result2
        serialized['comparison_id'] = self.comparison
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'benchmark_result1_id' in serialized_representation:
            kwargs['benchmark_result1_id'] = serialized_representation['benchmark_result1_id']
        if 'benchmark_result2_id' in serialized_representation:
            kwargs['benchmark_result2_id'] = serialized_representation['benchmark_result2_id']
        if 'comparison_id' in serialized_representation:
            kwargs['comparison_id'] = serialized_representation['comparison_id']
        return super().deserialize(serialized_representation, db_client, **kwargs)
