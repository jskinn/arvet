# Copyright (c) 2017, John Skinner
import unittest
import arvet.core.trial_result
import arvet.core.benchmark
import arvet.core.sequence_type
import arvet.core.tests.mock_types as mock_core
import arvet.database.tests.mock_database_client as mock_client_factory
import arvet.database.tests.test_entity
import arvet.database.client
import arvet.util.dict_utils as du
import arvet.batch_analysis.simple_experiment as ex


# A minimal experiment, so we can instantiate the abstract class
class MockExperiment(ex.SimpleExperiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_imports(self, task_manager, path_manager, db_client):
        pass


class TestExperiment(unittest.TestCase, arvet.database.tests.test_entity.EntityContract):

    def setUp(self):
        # Some complete results, so that we can have a non-empty trial map and result map on entity tests
        self.systems = [mock_core.MockSystem()]
        self.image_sources = [mock_core.MockImageSource() for _ in range(2)]
        self.trial_results = [arvet.core.trial_result.TrialResult(
                    system.identifier, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})
            for _ in self.image_sources for system in self.systems]
        self.benchmarks = [mock_core.MockBenchmark() for _ in range(2)]
        self.benchmark_results = [arvet.core.benchmark.BenchmarkResult(benchmark.identifier, trial_result.identifier,
                                                                       True)
                                  for benchmark in self.benchmarks for trial_result in self.trial_results]

    def get_class(self):
        return MockExperiment

    def make_instance(self, *args, **kwargs):
        du.defaults(kwargs, {
            'systems': {str(idx): self.systems[idx] for idx in range(len(self.systems))},
            'datasets': {str(idx): self.image_sources[idx] for idx in range(len(self.image_sources))},
            'benchmarks': {str(idx): self.benchmarks[idx] for idx in range(len(self.benchmarks))},
            'trial_map': {system.identifier: {image_source.identifier: trial_result.identifier
                                              for trial_result in self.trial_results
                                              if trial_result.identifier is not None and
                                              trial_result.system_id == system.identifier
                                              for image_source in self.image_sources
                                              if image_source.identifier is not None}
                          for system in self.systems if system.identifier is not None},
            'result_map': {trial_result.identifier: {benchmark.identifier: benchmark_result.identifier
                                                     for benchmark in self.benchmarks
                                                     if benchmark.identifier is not None
                                                     for benchmark_result in self.benchmark_results
                                                     if benchmark_result.identifier is not None and
                                                     benchmark_result.benchmark == benchmark.identifier and
                                                     benchmark_result.trial_result == trial_result.identifier}
                           for trial_result in self.trial_results
                           if trial_result.identifier is not None}
        })
        return MockExperiment(*args, **kwargs)

    def assert_models_equal(self, task1, task2):
        """
        Helper to assert that two tasks are equal
        We're going to violate encapsulation for a bit
        :param task1:
        :param task2:
        :return:
        """
        if (not isinstance(task1, ex.SimpleExperiment) or
                not isinstance(task2, ex.SimpleExperiment)):
            self.fail('object was not an Experiment')
        self.assertEqual(task1.identifier, task2.identifier)
        self.assertEqual(task1._trial_map, task2._trial_map)
        self.assertEqual(task1._result_map, task2._result_map)

    def create_mock_db_client(self):
        mock_db_client = super().create_mock_db_client()

        # Insert all the background data we expect.
        for system in self.systems:
            result = mock_db_client.system_collection.insert_one(system.serialize())
            system.refresh_id(result.inserted_id)
        for image_source in self.image_sources:
            result = mock_db_client.image_source_collection.insert_one(image_source.serialize())
            image_source.refresh_id(result.inserted_id)
        for trial_result in self.trial_results:
            result = mock_db_client.trials_collection.insert_one(trial_result.serialize())
            trial_result.refresh_id(result.inserted_id)
        for benchmark in self.benchmarks:
            result = mock_db_client.benchmarks_collection.insert_one(benchmark.serialize())
            benchmark.refresh_id(result.inserted_id)
        for benchmark_result in self.benchmark_results:
            result = mock_db_client.results_collection.insert_one(benchmark_result.serialize())
            benchmark_result.refresh_id(result.inserted_id)

        return mock_db_client

    def test_constructor_works_with_minimal_arguments(self):
        MockExperiment()
