# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import bson
import pymongo.collection
import arvet.database.tests.test_entity
import arvet.util.dict_utils as du
import arvet.batch_analysis.task as task


class TestTask(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return task.Task

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'state': task.JobState.RUNNING,
            'num_cpus': np.random.randint(0, 1000),
            'num_gpus': np.random.randint(0, 1000),
            'memory_requirements': '{}MB'.format(np.random.randint(0, 50000)),
            'expected_duration': '{0}:{1}:{2}'.format(np.random.randint(1000), np.random.randint(60),
                                                      np.random.randint(60)),
            'node_id': 'node-{}'.format(np.random.randint(10000)),
            'job_id': np.random.randint(1000)
        })
        return task.Task(*args, **kwargs)

    def assert_models_equal(self, task1, task2):
        """
        Helper to assert that two tasks are equal
        We're going to violate encapsulation for a bit
        :param task1:
        :param task2:
        :return:
        """
        if (not isinstance(task1, task.Task) or
                not isinstance(task2, task.Task)):
            self.fail('object was not an Task')
        self.assertEqual(task1.identifier, task2.identifier)
        self.assertEqual(task1._state, task2._state)
        self.assertEqual(task1.node_id, task2.node_id)
        self.assertEqual(task1.job_id, task2.job_id)
        self.assertEqual(task1.result, task2.result)
        self.assertEqual(task1.num_cpus, task2.num_cpus)
        self.assertEqual(task1.num_gpus, task2.num_gpus)
        self.assertEqual(task1.memory_requirements, task2.memory_requirements)
        self.assertEqual(task1.expected_duration, task2.expected_duration)

    def test_mark_job_started_changes_unstarted_to_running(self):
        subject = task.Task(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        subject.mark_job_started('test', 12)
        self.assertFalse(subject.is_unstarted)

    def test_mark_job_started_doesnt_affect_already_running_jobs(self):
        subject = task.Task(state=task.JobState.RUNNING, node_id='external', job_id=3)
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        subject.mark_job_started('test', 12)
        self.assertFalse(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        self.assertEqual('external', subject.node_id)
        self.assertEqual(3, subject.job_id)
        self.assertEqual({}, subject._updates)

    def test_mark_job_started_doesnt_affect_finished_jobs(self):
        result_id = bson.ObjectId()
        subject = task.Task(state=task.JobState.DONE, result=result_id)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_finished)
        subject.mark_job_started('test', 12)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)
        self.assertEqual(result_id, subject.result)
        self.assertEqual({}, subject._updates)

    def test_mark_job_started_stores_updates(self):
        subject = task.Task(state=task.JobState.UNSTARTED, id_=bson.ObjectId())
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        subject.mark_job_started('test', 12)
        subject.save_updates(mock_collection)
        self.assertTrue(mock_collection.update.called)
        self.assertEqual({'_id': subject.identifier}, mock_collection.update.call_args[0][0])
        query = mock_collection.update.call_args[0][1]
        self.assertIn('$set', query)
        self.assertIn('node_id', query['$set'])
        self.assertIn('job_id', query['$set'])

    def test_mark_job_failed_changes_running_to_unstarted(self):
        subject = task.Task(state=task.JobState.RUNNING, node_id='test', job_id=5)
        self.assertFalse(subject.is_unstarted)
        subject.mark_job_failed()
        self.assertTrue(subject.is_unstarted)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)

    def test_mark_job_failed_doesnt_affect_unstarted_jobs(self):
        subject = task.Task(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        subject.mark_job_failed()
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        self.assertEqual({}, subject._updates)

    def test_mark_job_failed_doesnt_affect_finished_jobs(self):
        result_id = bson.ObjectId()
        subject = task.Task(state=task.JobState.DONE, result=result_id)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_finished)
        subject.mark_job_failed()
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)
        self.assertEqual(result_id, subject.result)
        self.assertEqual({}, subject._updates)

    def test_mark_job_failed_stores_updates(self):
        subject = task.Task(state=task.JobState.RUNNING, node_id='test', job_id=14, id_=bson.ObjectId())
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        subject.mark_job_failed()
        subject.save_updates(mock_collection)
        self.assertTrue(mock_collection.update.called)
        self.assertEqual({'_id': subject.identifier}, mock_collection.update.call_args[0][0])
        query = mock_collection.update.call_args[0][1]
        self.assertIn('$unset', query)
        self.assertIn('node_id', query['$unset'])
        self.assertIn('job_id', query['$unset'])

    def test_mark_job_complete_changes_running_to_finished(self):
        result_id = bson.ObjectId()
        subject = task.Task(state=task.JobState.RUNNING, node_id='test', job_id=5)
        self.assertFalse(subject.is_unstarted)
        subject.mark_job_complete(result_id)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)
        self.assertEqual(result_id, subject.result)

    def test_mark_job_complete_doesnt_affect_unstarted_jobs(self):
        subject = task.Task(state=task.JobState.UNSTARTED)
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        subject.mark_job_complete(bson.ObjectId())
        self.assertTrue(subject.is_unstarted)
        self.assertFalse(subject.is_finished)
        self.assertEqual({}, subject._updates)
        self.assertIsNone(subject.result)

    def test_mark_job_complete_doesnt_affect_finished_jobs(self):
        result_id = bson.ObjectId()
        subject = task.Task(state=task.JobState.DONE, result=result_id)
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_finished)
        subject.mark_job_complete(bson.ObjectId())
        self.assertFalse(subject.is_unstarted)
        self.assertTrue(subject.is_finished)
        self.assertIsNone(subject.node_id)
        self.assertIsNone(subject.job_id)
        self.assertEqual(result_id, subject.result)
        self.assertEqual({}, subject._updates)

    def test_mark_job_complete_stores_updates(self):
        result_id = bson.ObjectId()
        subject = task.Task(state=task.JobState.RUNNING, node_id='test', job_id=14, id_=bson.ObjectId())
        mock_collection = mock.create_autospec(pymongo.collection.Collection)
        subject.mark_job_complete(result_id)
        subject.save_updates(mock_collection)
        self.assertTrue(mock_collection.update.called)
        self.assertEqual({'_id': subject.identifier}, mock_collection.update.call_args[0][0])
        query = mock_collection.update.call_args[0][1]
        self.assertIn('$set', query)
        self.assertIn('result', query['$set'])
        self.assertIn('$unset', query)
        self.assertIn('node_id', query['$unset'])
        self.assertIn('job_id', query['$unset'])

    def test_state_remains_consistent(self):
        random = np.random.RandomState(144135)
        subject = task.Task(state=task.JobState.UNSTARTED)
        for idx in range(50):
            change = random.randint(0, 3 if idx > 30 else 2)
            if idx > 30 and change == 2:
                subject.mark_job_complete(bson.ObjectId())
            elif change == 1:
                subject.mark_job_started('test', random.randint(0, 1000))
            else:
                subject.mark_job_failed()
            # Make sure that the node id, job id, and result match the state
            if subject.is_unstarted:
                self.assertIsNone(subject.node_id)
                self.assertIsNone(subject.job_id)
                self.assertIsNone(subject.result)
            elif subject.is_finished:
                self.assertIsNone(subject.node_id)
                self.assertIsNone(subject.job_id)
                self.assertIsNotNone(subject.result)
            else:
                self.assertIsNotNone(subject.node_id)
                self.assertIsNotNone(subject.job_id)
                self.assertIsNone(subject.result)
            # Make sure we don't have an empty update, and we don't set and unset the same keys
            set_keys = set()
            unset_keys = set()
            if '$set' in subject._updates:
                self.assertNotEqual({}, subject._updates['$set'])
                set_keys = set(subject._updates['$set'].keys())
            if '$unset' in subject._updates:
                self.assertNotEqual({}, subject._updates['$unset'])
                unset_keys = set(subject._updates['$unset'].keys())
            self.assertEqual(set(), set_keys & unset_keys)
