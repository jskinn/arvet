# Copyright (c) 2017, John Skinner
import typing
import bson
import arvet.batch_analysis.task
import arvet.database.client
import arvet.config.path_manager


class BenchmarkTrialTask(arvet.batch_analysis.task.Task):
    """
    A task for benchmarking a trial result. Result is a BenchmarkResult id.
    """
    def __init__(self, trial_result_ids: typing.Iterable[bson.ObjectId], benchmark_id: bson.ObjectId, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial_result_ids = set(trial_result_ids)
        self._benchmark_id = benchmark_id

    @property
    def trial_results(self) -> typing.Set[bson.ObjectId]:
        return self._trial_result_ids

    @property
    def benchmark(self) -> bson.ObjectId:
        return self._benchmark_id

    def run_task(self, path_manager: arvet.config.path_manager.PathManager,
                 db_client: arvet.database.client.DatabaseClient) -> None:
        import logging
        import traceback
        import arvet.util.database_helpers as dh

        benchmark = dh.load_object(db_client, db_client.benchmarks_collection, self.benchmark)
        trial_results = dh.load_many_objects(db_client, db_client.trials_collection, self.trial_results)

        if len(trial_results) < len(self.trial_results):
            logging.getLogger(__name__).error("Could not deserialize trial results {0}".format(
                set(self.trial_results) - set(trial_result.identifier for trial_result in trial_results)
            ))
            self.mark_job_failed()
            return
        elif benchmark is None:
            logging.getLogger(__name__).error("Could not deserialize benchmark {0}".format(self.benchmark))
            self.mark_job_failed()
            return

        # Check all the trials are appropriate
        for trial_result in trial_results:
            if not benchmark.is_trial_appropriate(trial_result):
                logging.getLogger(__name__).error("Benchmark {0} cannot assess trial {1}".format(
                    self.benchmark, trial_result.identifier))
                self.mark_job_failed()
                return

        logging.getLogger(__name__).info("Benchmarking results {0} with benchmark {1}".format(self.trial_results,
                                                                                              self.benchmark))
        try:
            benchmark_result = benchmark.benchmark_results(trial_results)
        except Exception as exception:
            logging.getLogger(__name__).error("Exception while benchmarking {0} with benchmark {1}:\n{2}".format(
                self.trial_results, self.benchmark, traceback.format_exc()))
            self.mark_job_failed()
            raise exception  # Re-raise the caught exception
        if benchmark_result is None:
            logging.getLogger(__name__).error("Failed to benchmark {0} with {1}".format(
                self.trial_results, self.benchmark))
            self.mark_job_failed()
        else:
            benchmark_result_id = db_client.results_collection.insert(benchmark_result.serialize())
            logging.getLogger(__name__).info("Successfully benchmarked trial {0} with benchmark {1},"
                                             "producing result {2}".format(self.trial_results, self.benchmark,
                                                                           benchmark_result_id))
            self.mark_job_complete(benchmark_result_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['trial_result_ids'] = list(self.trial_results)
        serialized['benchmark_id'] = self.benchmark
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trial_result_ids' in serialized_representation:
            kwargs['trial_result_ids'] = serialized_representation['trial_result_ids']
        if 'benchmark_id' in serialized_representation:
            kwargs['benchmark_id'] = serialized_representation['benchmark_id']
        return super().deserialize(serialized_representation, db_client, **kwargs)
