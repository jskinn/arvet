# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import arvet.database.tests.database_connection as dbconn
from arvet.config.path_manager import PathManager
import arvet.batch_analysis.task as task


class MockTask(task.Task):
    def run_task(self, path_manager: PathManager):
        pass

    def get_unique_name(self) -> str:
        return "mock_task_{0}".format(self.pk)


class TestTaskDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        task.Task._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        task.Task._mongometa.collection.drop()

    def test_stores_and_loads_simple(self):
        obj = MockTask(state=task.JobState.UNSTARTED)
        obj.save()

        # Load all the entities
        all_entities = list(task.Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_all_params(self):
        obj = MockTask(
            state=task.JobState.RUNNING,
            node_id='test-hpc',
            job_id=15,
            num_cpus=3,
            num_gpus=150,
            memory_requirements='4KB',
            expected_duration='100:00:00'
        )
        obj.save()

        # Load all the entities
        all_entities = list(task.Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_after_change_state(self):
        obj = MockTask(
            state=task.JobState.RUNNING,
            node_id='test-hpc',
            job_id=15,
            num_cpus=3,
            num_gpus=150,
            memory_requirements='4KB',
            expected_duration='100:00:00'
        )
        obj.save()

        all_entities = list(task.Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)

        obj.mark_job_failed()
        obj.save()

        all_entities = list(task.Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)

        obj.mark_job_started('test_node', 143)
        obj.save()

        all_entities = list(task.Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)

        obj.mark_job_complete()
        obj.save()

        all_entities = list(task.Task.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)


class TestTask(unittest.TestCase):

    def test_mark_job_started_changes_unstarted_to_running(self):
        subject = MockTask(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        subject.mark_job_started('test', 12)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)

    def test_mark_job_started_doesnt_affect_already_running_jobs(self):
        subject = MockTask(state=task.JobState.RUNNING, node_id='external', job_id=3)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertFalse(subject.is_finished)
        subject.mark_job_started('test', 12)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertFalse(subject.is_finished)
        self.assertEqual('external', subject.node_id)
        self.assertEqual(3, subject.job_id)

    def test_mark_job_started_doesnt_affect_finished_jobs(self):
        subject = MockTask(state=task.JobState.DONE)
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        subject.mark_job_started('test', 12)
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_mark_job_failed_changes_running_to_unstarted(self):
        subject = MockTask(state=task.JobState.RUNNING, node_id='test', job_id=5)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        subject.mark_job_failed()
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_mark_job_failed_increases_failed_count(self):
        subject = MockTask(state=task.JobState.RUNNING, node_id='test', job_id=5, failure_count=4)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        subject.mark_job_failed()
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertEqual(5, subject.failure_count)

    def test_mark_job_failed_doesnt_affect_unstarted_jobs(self):
        subject = MockTask(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        subject.mark_job_failed()
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        self.assertEqual(0, subject.failure_count)

    def test_mark_job_failed_doesnt_affect_finished_jobs(self):
        subject = MockTask(state=task.JobState.DONE)
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        subject.mark_job_failed()
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_mark_job_complete_changes_running_to_finished(self):
        subject = MockTask(state=task.JobState.RUNNING, node_id='test', job_id=5)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        subject.mark_job_complete()
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_mark_job_complete_doesnt_affect_unstarted_jobs(self):
        subject = MockTask(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)
        subject.mark_job_complete()
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_mark_job_complete_doesnt_affect_finished_jobs(self):
        subject = MockTask(state=task.JobState.DONE)
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        subject.mark_job_complete()
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_running)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_change_job_id_doesnt_affect_state(self):
        subject = MockTask(state=task.JobState.RUNNING)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertFalse(subject.is_finished)
        subject.change_job_id('test', 12)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_running)
        self.assertFalse(subject.is_finished)

    def test_change_job_id_changes_job_info(self):
        subject = MockTask(state=task.JobState.RUNNING, node_id='external', job_id=3)
        self.assertEqual('external', subject.node_id)
        self.assertEqual(3, subject.job_id)
        subject.change_job_id('test', 12)
        self.assertEqual('test', subject.node_id)
        self.assertEqual(12, subject.job_id)

    def test_change_job_id_doesnt_affect_unstarted_jobs(self):
        subject = MockTask(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        subject.change_job_id('test', 12)
        self.assertTrue(subject.is_unstarted)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_change_job_id_doesnt_affect_finished_jobs(self):
        subject = MockTask(state=task.JobState.DONE, node_id='external', job_id=3)
        self.assertTrue(subject.is_finished)
        self.assertEqual('external', subject.node_id)
        self.assertEqual(3, subject.job_id)
        subject.change_job_id('test', 12)
        self.assertTrue(subject.is_finished)
        self.assertEqual('external', subject.node_id)
        self.assertEqual(3, subject.job_id)

    def test_state_remains_consistent(self):
        random = np.random.RandomState(144135)
        subject = MockTask(state=task.JobState.UNSTARTED)
        for idx in range(50):
            change = random.randint(0, 4 if idx > 30 else 3)
            if idx > 30 and change == 3:
                subject.mark_job_complete()
            elif change == 2:
                subject.change_job_id('external', random.randint(0, 1000))
            elif change == 1:
                subject.mark_job_started('test', random.randint(0, 1000))
            else:
                subject.mark_job_failed()
            # Make sure that the node id and job id match the state
            if subject.is_unstarted or subject.is_finished:
                self.assertIsNone(subject.node_id)
                self.assertIsNone(subject.job_id)
            else:
                self.assertIsNotNone(subject.node_id)
                self.assertIsNotNone(subject.job_id)
