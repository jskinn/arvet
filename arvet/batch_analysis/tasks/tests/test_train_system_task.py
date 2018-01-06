# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import bson
import arvet.database.tests.test_entity
import arvet.util.dict_utils as du
import arvet.batch_analysis.task
import arvet.batch_analysis.tasks.train_system_task as task


class TestTrainSystemTask(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return task.TrainSystemTask

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'trainer_id': bson.ObjectId(),
            'trainee_id': bson.ObjectId(),
            'state': arvet.batch_analysis.task.JobState.RUNNING,
            'num_cpus': np.random.randint(0, 1000),
            'num_gpus': np.random.randint(0, 1000),
            'memory_requirements': '{}MB'.format(np.random.randint(0, 50000)),
            'expected_duration': '{0}:{1}:{2}'.format(np.random.randint(1000), np.random.randint(60),
                                                      np.random.randint(60)),
            'node_id': 'node-{}'.format(np.random.randint(10000)),
            'job_id': np.random.randint(1000)
        })
        return task.TrainSystemTask(*args, **kwargs)

    def assert_models_equal(self, task1, task2):
        """
        Helper to assert that two tasks are equal
        We're going to violate encapsulation for a bit
        :param task1:
        :param task2:
        :return:
        """
        if (not isinstance(task1, task.TrainSystemTask) or
                not isinstance(task2, task.TrainSystemTask)):
            self.fail('object was not an TrainSystemTask')
        self.assertEqual(task1.identifier, task2.identifier)
        self.assertEqual(task1.trainer, task2.trainer)
        self.assertEqual(task1.trainee, task2.trainee)
        self.assertEqual(task1._state, task2._state)
        self.assertEqual(task1.node_id, task2.node_id)
        self.assertEqual(task1.job_id, task2.job_id)
        self.assertEqual(task1.result, task2.result)
        self.assertEqual(task1.num_cpus, task2.num_cpus)
        self.assertEqual(task1.num_gpus, task2.num_gpus)
        self.assertEqual(task1.memory_requirements, task2.memory_requirements)
        self.assertEqual(task1.expected_duration, task2.expected_duration)
