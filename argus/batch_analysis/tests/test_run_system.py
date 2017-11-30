# Copyright (c) 2017, John Skinner
import unittest, unittest.mock as mock
import argus.database.client
import argus.core.system


class TestRunSystem(unittest.TestCase):

    def setUp(self):

        self._mock_vision_system_patch = mock.patch('argus.core.benchmark.Benchmark', spec=argus.core.system.VisionSystem)
        self._mock_vision_system = self._mock_vision_system_patch.start()

        self._mock_db_client_patch = mock.patch('argus.database.client.DatabaseClient', spec=argus.database.client.DatabaseClient)
        self._mock_db_client = self._mock_db_client_patch.start()

        self.existing_datasets = [{'trial_result': 1},{'trial_result': 2},{'trial_result': 3}]
        self.trial_results = [{
            '_type': 'Test'
        },{
            '_type': 'Test'
        }]

        self.vision_system = self._mock_vision_system()
        self.vision_system.get_benchmark_requirements = mock.Mock(return_value={})

        self.db_client = self._mock_db_client()
        self.db_client.dataset_collection = mock.Mock()
        self.db_client.dataset_collection.find = mock.Mock(return_value=self.existing_datasets)
        self.db_client.trials_collection = mock.Mock()
        self.db_client.trials_collection.find = mock.Mock(return_value=self.trial_results)
        self.db_client.deserialize_entity = mock_deserialize_entity

    def tearDown(self):
        self._mock_vision_system_patch.stop()
        self._mock_db_client_patch.stop()

    # TODO: This needs either extensive mocking, or full integration tests.
    #def test_gets_existing_results(self, MockVisionSystem, MockDatabaseClient):
    #    test_vision_system(self.vision_system, self.db_client)


def mock_deserialize_entity(s_entity):
    pass