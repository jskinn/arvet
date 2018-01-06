# Copyright (c) 2017, John Skinner
import unittest, unittest.mock as mock
import arvet.core.benchmark
import arvet.database.client
import arvet.batch_analysis.benchmark_results as bench


class TestBenchmarkResults(unittest.TestCase):
    """
    This is a test for mocking the parent modules, it doesn't work.
    Re-write as batch analysis is updated

    def setUp(self):

        self._mock_benchmark_patch = mock.patch('arvet.core.benchmark.Benchmark', spec=arvet.core.benchmark.Benchmark)
        self._mock_benchmark = self._mock_benchmark_patch.start()

        self._mock_db_client_patch = mock.patch('arvet.database.client.DatabaseClient', spec=arvet.database.client.DatabaseClient)
        self._mock_db_client = self._mock_db_client_patch.start()

        self.existing_results = [{'trial_result': 1}, {'trial_result': 2}, {'trial_result': 3}]
        self.trial_results = [{
            '_type': 'Test'
        },{
            '_type': 'Test'
        }]

        self.benchmark = self._mock_benchmark()
        self.benchmark.get_trial_requirements = mock.Mock(return_value={})

        self.db_client = self._mock_db_client()
        self.db_client.results_collection = mock.Mock()
        self.db_client.results_collection.find = mock.Mock(return_value=self.existing_results)
        self.db_client.trials_collection = mock.Mock()
        self.db_client.trials_collection.find = mock.Mock(return_value=self.trial_results)

    def tearDown(self):
        self._mock_benchmark_patch.stop()
        self._mock_db_client_patch.stop()

    def test_gets_existing_results(self):
        bench.benchmark_results(self.benchmark, self.db_client, {})
        self.assertTrue(self.db_client.results_collection.find.called)

    def test_excludes_existing_results(self):
        bench.benchmark_results(self.benchmark, self.db_client, {})

        self.assertTrue(self.db_client.trials_collection.find.called)
        call_args, call_kwargs = self.db_client.trials_collection.find.call_args
        query = call_kwargs['filter'] if 'filter' in call_kwargs else call_args[0]
        self.assertEqual(query['_id'], {'$nin': [1, 2, 3]})
    """
    pass
