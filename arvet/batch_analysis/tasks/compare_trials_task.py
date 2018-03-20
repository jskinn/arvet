# Copyright (c) 2017, John Skinner
import typing
import bson
import arvet.batch_analysis.task
import arvet.database.client
import arvet.config.path_manager


class CompareTrialTask(arvet.batch_analysis.task.Task):
    """
    A task for comparing two trial results against each other. Result is a TrialComparison id
    """
    def __init__(self, trial_result_ids: typing.Iterable[bson.ObjectId],
                 reference_trial_result_ids: typing.Iterable[bson.ObjectId],
                 comparison_id: bson.ObjectId, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial_result_ids = set(trial_result_ids)
        self._reference_trial_result_ids = set(reference_trial_result_ids)
        self._comparison_id = comparison_id

    @property
    def trial_results(self) -> typing.Set[bson.ObjectId]:
        return self._trial_result_ids

    @property
    def reference_trial_results(self) -> typing.Set[bson.ObjectId]:
        return self._reference_trial_result_ids

    @property
    def comparison(self) -> bson.ObjectId:
        return self._comparison_id

    def run_task(self, path_manager: arvet.config.path_manager.PathManager,
                 db_client: arvet.database.client.DatabaseClient) -> None:
        import logging
        import traceback
        import arvet.util.database_helpers as dh

        trial_results = dh.load_many_objects(db_client, db_client.trials_collection, self.trial_results)
        reference_trial_results = dh.load_many_objects(db_client, db_client.trials_collection,
                                                       self.reference_trial_results)
        comparison_benchmark = dh.load_object(db_client, db_client.benchmarks_collection, self.comparison)

        if len(trial_results) < len(self.trial_results) is None:
            logging.getLogger(__name__).error("Could not deserialize {0} trial results".format(
                len(self.trial_results) - len(trial_results)))
            self.mark_job_failed()
        elif len(reference_trial_results) < len(self.reference_trial_results):
            logging.getLogger(__name__).error("Could not deserialize {0} reference trial results".format(
                len(self.reference_trial_results) - len(reference_trial_results)))
            self.mark_job_failed()
        elif comparison_benchmark is None:
            logging.getLogger(__name__).error("Could not deserialize comparison benchmark {0}".format(self.comparison))
            self.mark_job_failed()
        elif (not comparison_benchmark.is_trial_appropriate(trial_results) or
                not comparison_benchmark.is_trial_appropriate(reference_trial_results)):
            logging.getLogger(__name__).error("Benchmark {0} is not appropriate for trial {1} or {2}".format(
                self.comparison, self.trial_result1, self.trial_result2))
            self.mark_job_failed()
        else:
            logging.getLogger(__name__).info("Comparing trial results {0} and {1} with comparison benchmark {2}".format(
                self.trial_results, self.reference_trial_results, self.comparison))
            try:
                comparison_result = comparison_benchmark.compare_trial_results(trial_results, reference_trial_results)
            except Exception as exception:
                logging.getLogger(__name__).error(
                    "Error occurred while comparing trials {0} and {1} with benchmark {2}:\n{3}".format(
                        self.trial_results, self.reference_trial_results, self.comparison, traceback.format_exc()))
                self.mark_job_failed()
                raise exception
            if comparison_result is None:
                logging.getLogger(__name__).error(
                    "Failed to compare trials {0} and {1} with benchmark {2}".format(
                        self.trial_results, self.reference_trial_results, self.comparison))
                self.mark_job_failed()
            else:
                comparison_result.save_data(db_client)
                result_id = db_client.results_collection.insert(comparison_result.serialize())
                logging.getLogger(__name__).info("Successfully compared trials {0} and {1},"
                                                 "producing result {2}".format(self.trial_results,
                                                                               self.reference_trial_results,
                                                                               result_id))
                self.mark_job_complete(result_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['trial_result_ids'] = list(self.trial_results)
        serialized['reference_trial_result_ids'] = list(self.reference_trial_results)
        serialized['comparison_id'] = self.comparison
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trial_result_ids' in serialized_representation:
            kwargs['trial_result_ids'] = serialized_representation['trial_result_ids']
        if 'reference_trial_result_ids' in serialized_representation:
            kwargs['reference_trial_result_ids'] = serialized_representation['reference_trial_result_ids']
        if 'comparison_id' in serialized_representation:
            kwargs['comparison_id'] = serialized_representation['comparison_id']
        return super().deserialize(serialized_representation, db_client, **kwargs)
