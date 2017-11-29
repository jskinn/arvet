# Copyright (c) 2017, John Skinner
import batch_analysis.task


class BenchmarkTrialTask(batch_analysis.task.Task):
    """
    A task for benchmarking a trial result. Result is a BenchmarkResult id.
    """
    def __init__(self, trial_result_id, benchmark_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial_result_id = trial_result_id
        self._benchmark_id = benchmark_id

    @property
    def trial_result(self):
        return self._trial_result_id

    @property
    def benchmark(self):
        return self._benchmark_id

    def run_task(self, db_client):
        import logging
        import traceback
        import util.database_helpers as dh

        trial_result = dh.load_object(db_client, db_client.trials_collection, self.trial_result)
        benchmark = dh.load_object(db_client, db_client.benchmarks_collection, self.benchmark)

        if trial_result is None:
            logging.getLogger(__name__).error("Could not deserialize trial result {0}".format(self.trial_result))
            self.mark_job_failed()
        elif benchmark is None:
            logging.getLogger(__name__).error("Could not deserialize benchmark {0}".format(self.benchmark))
            self.mark_job_failed()
        elif not benchmark.is_trial_appropriate(trial_result):
            logging.getLogger(__name__).error("Benchmark {0} cannot assess trial {1}".format(
                self.benchmark, self.trial_result))
            self.mark_job_failed()
        else:
            logging.getLogger(__name__).info("Benchmarking result {0} with benchmark {1}".format(self.trial_result,
                                                                                                 self.benchmark))
            try:
                benchmark_result = benchmark.benchmark_results(trial_result)
            except Exception:
                logging.getLogger(__name__).error("Exception while benchmarking {0} with benchmark {1}:\n{2}".format(
                    self.trial_result, self.benchmark, traceback.format_exc()))
                benchmark_result = None
            if benchmark_result is None:
                logging.getLogger(__name__).error("Failed to benchmark {0} with {1}".format(
                    self.trial_result, self.benchmark))
                self.mark_job_failed()
            else:
                benchmark_result_id = db_client.results_collection.insert(benchmark_result.serialize())
                logging.getLogger(__name__).info("Successfully benchmarked trial {0} with benchmark {1},"
                                                 "producing result {2}".format(self.trial_result, self.benchmark,
                                                                               benchmark_result_id))
                self.mark_job_complete(benchmark_result_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['trial_result_id'] = self.trial_result
        serialized['benchmark_id'] = self.benchmark
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trial_result_id' in serialized_representation:
            kwargs['trial_result_id'] = serialized_representation['trial_result_id']
        if 'benchmark_id' in serialized_representation:
            kwargs['benchmark_id'] = serialized_representation['benchmark_id']
        return super().deserialize(serialized_representation, db_client, **kwargs)
