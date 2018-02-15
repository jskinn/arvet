import unittest
import unittest.mock as mock
import bson
import arvet.database.tests.mock_database_client
import arvet.core.image_collection
import arvet.batch_analysis.task_manager
import arvet.batch_analysis.job_system
import arvet.batch_analysis.scripts.warmup_image_cache as warmup_cache_script


class TestWarmupCache(unittest.TestCase):

    def test_warmup_cache_warms_cache_for_all_image_collections(self):
        image_collections = {
            bson.ObjectId(): mock.create_autospec(arvet.core.image_collection.ImageCollection, spec_set=True)
            for _ in range(3)
        }
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        mock_db_client.deserialize_entity.side_effect = lambda s_obj: image_collections[s_obj['_id']]
        mock_db_client.image_source_collection.find.return_value = [{'_id': oid} for oid in image_collections.keys()]

        with mock.patch('arvet.config.global_configuration.load_global_config', autospec=True) as mock_conf:
            mock_conf.return_value = {}
            with mock.patch('arvet.database.client.DatabaseClient', autospec=True) as mock_database_constructor:
                mock_database_constructor.return_value = mock_db_client
                warmup_cache_script.warmup_cache([str(oid) for oid in image_collections.keys()], [])

        for mock_image_collection in image_collections.values():
            self.assertTrue(mock_image_collection.warmup_cache.called)

    def test_warmup_cache_runs_dependent_tasks(self):
        task_ids = [bson.ObjectId() for _ in range(10)]
        mock_db_client = arvet.database.tests.mock_database_client.create().mock
        mock_task_manager = mock.create_autospec(arvet.batch_analysis.task_manager.TaskManager, spec_set=True)
        mock_job_system = mock.create_autospec(arvet.batch_analysis.job_system.JobSystem, spec_set=True)

        # A whole pile of dependency injection to control the objects that get created
        with mock.patch('arvet.config.global_configuration.load_global_config', autospec=True) as mock_conf:
            mock_conf.return_value = {}
            with mock.patch('arvet.database.client.DatabaseClient', autospec=True) as mock_database_constructor:
                mock_database_constructor.return_value = mock_db_client
                with mock.patch('arvet.batch_analysis.task_manager.TaskManager', autospec=True) as mock_tm:
                    mock_tm.return_value = mock_task_manager
                    with mock.patch('arvet.batch_analysis.job_systems.job_system_factory.create_job_system',
                                    autospec=True) as mock_jf:
                        mock_jf.return_value = mock_job_system
                        warmup_cache_script.warmup_cache([], task_ids)

        self.assertEqual(mock.call(task_ids, mock_job_system), mock_task_manager.schedule_dependent_tasks.call_args)
        self.assertTrue(mock_job_system.run_queued_jobs.called)

    def test_main_parses_arguments(self):
        image_collection_id = bson.ObjectId()
        task_ids = [bson.ObjectId() for _ in range(10)]
        args = [warmup_cache_script.__file__, '--image_collection', str(image_collection_id)] + \
               [str(oid) for oid in task_ids]
        with mock.patch('arvet.batch_analysis.scripts.warmup_image_cache.warmup_cache', autospec=True) as mock_warm:
            with mock.patch('sys.argv', args):
                warmup_cache_script.main()
            self.assertEqual(1, mock_warm.call_count)
            self.assertEqual(mock.call(
                [str(image_collection_id)],
                [str(oid) for oid in task_ids]
            ), mock_warm.call_args)
