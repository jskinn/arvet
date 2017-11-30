# Copyright (c) 2017, John Skinner
import argus.batch_analysis.task


class CompareTrialTask(argus.batch_analysis.task.Task):
    """
    A task for comparing two trial results against each other. Result is a TrialComparison id
    """
    def __init__(self, trial_result1_id, trial_result2_id, comparison_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trial_result1_id = trial_result1_id
        self._trial_result2_id = trial_result2_id
        self._comparison_id = comparison_id

    @property
    def trial_result1(self):
        return self._trial_result1_id

    @property
    def trial_result2(self):
        return self._trial_result2_id

    @property
    def comparison(self):
        return self._comparison_id

    def run_task(self, db_client):
        import logging
        import traceback
        import argus.util.database_helpers as dh

        trial_result_1 = dh.load_object(db_client, db_client.trials_collection, self.trial_result1)
        trial_result_2 = dh.load_object(db_client, db_client.trials_collection, self.trial_result2)
        comparison_benchmark = dh.load_object(db_client, db_client.benchmarks_collection, self.comparison)

        if trial_result_1 is None:
            logging.getLogger(__name__).error("Could not deserialize trial result {0}".format(self.trial_result1))
            self.mark_job_failed()
        elif trial_result_2 is None:
            logging.getLogger(__name__).error("Could not deserialize trial result {0}".format(self.trial_result2))
            self.mark_job_failed()
        elif comparison_benchmark is None:
            logging.getLogger(__name__).error("Could not deserialize comparison benchmark {0}".format(self.comparison))
            self.mark_job_failed()
        elif (not comparison_benchmark.is_trial_appropriate(trial_result_1) or
                not comparison_benchmark.is_trial_appropriate(trial_result_2)):
            logging.getLogger(__name__).error("Benchmark {0} is not appropriate for trial {1} or {2}".format(
                self.comparison, self.trial_result1, self.trial_result2))
            self.mark_job_failed()
        else:
            logging.getLogger(__name__).info("Comparing trial results {0} and {1} with comparison benchmark {2}".format(
                self.trial_result1, self.trial_result2, self.comparison))
            try:
                comparison_result = comparison_benchmark.compare_trial_results(trial_result_1, trial_result_2)
            except Exception:
                logging.getLogger(__name__).error("Error occurred while comparing trials {0} and {1}"
                                                  "with benchmark {2}:\n{3}".format(
                    self.trial_result1, self.trial_result2, self.comparison, traceback.format_exc()))
                comparison_result = None
            if comparison_result is None:
                logging.getLogger(__name__).error("Failed to compare trials {0} and {1}"
                                                  "with benchmark {2}".format(
                    self.trial_result1, self.trial_result2, self.comparison))
                self.mark_job_failed()
            else:
                comparison_result.save_data(db_client)
                result_id = db_client.results_collection.insert(comparison_result.serialize())
                logging.getLogger(__name__).info("Successfully compared trials {0} and {1},"
                                                 "producing result {2}".format(self.trial_result1, self.trial_result2,
                                                                               result_id))
                self.mark_job_complete(result_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['trial_result1_id'] = self.trial_result1
        serialized['trial_result2_id'] = self.trial_result2
        serialized['comparison_id'] = self.comparison
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trial_result1_id' in serialized_representation:
            kwargs['trial_result1_id'] = serialized_representation['trial_result1_id']
        if 'trial_result2_id' in serialized_representation:
            kwargs['trial_result2_id'] = serialized_representation['trial_result2_id']
        if 'comparison_id' in serialized_representation:
            kwargs['comparison_id'] = serialized_representation['comparison_id']
        return super().deserialize(serialized_representation, db_client, **kwargs)
