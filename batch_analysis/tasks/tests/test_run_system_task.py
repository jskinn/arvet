import unittest
import unittest.mock as mock
import numpy as np
import bson
import database.tests.test_entity
import core.image_source
import core.system
import util.dict_utils as du
import batch_analysis.task
import batch_analysis.tasks.run_system_task as task


class TestRunSystemTask(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return task.RunSystemTask

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'system_id': bson.ObjectId(),
            'image_source_id': bson.ObjectId(),
            'repeat': np.random.randint(0, 1000),
            'state': batch_analysis.task.JobState.RUNNING,
            'num_cpus': np.random.randint(0, 1000),
            'num_gpus': np.random.randint(0, 1000),
            'memory_requirements': '{}MB'.format(np.random.randint(0, 50000)),
            'expected_duration': '{0}:{1}:{2}'.format(np.random.randint(1000), np.random.randint(60),
                                                      np.random.randint(60)),
            'node_id': 'node-{}'.format(np.random.randint(10000)),
            'job_id': np.random.randint(1000)
        })
        return task.RunSystemTask(*args, **kwargs)

    def assert_models_equal(self, task1, task2):
        """
        Helper to assert that two tasks are equal
        We're going to violate encapsulation for a bit
        :param task1:
        :param task2:
        :return:
        """
        if (not isinstance(task1, task.RunSystemTask) or
                not isinstance(task2, task.RunSystemTask)):
            self.fail('object was not an RunSystemTask')
        self.assertEqual(task1.identifier, task2.identifier)
        self.assertEqual(task1.system, task2.system)
        self.assertEqual(task1.image_source, task2.image_source)
        self.assertEqual(task1._repeat, task2._repeat)
        self.assertEqual(task1._state, task2._state)
        self.assertEqual(task1.node_id, task2.node_id)
        self.assertEqual(task1.job_id, task2.job_id)
        self.assertEqual(task1.result, task2.result)
        self.assertEqual(task1.num_cpus, task2.num_cpus)
        self.assertEqual(task1.num_gpus, task2.num_gpus)
        self.assertEqual(task1.memory_requirements, task2.memory_requirements)
        self.assertEqual(task1.expected_duration, task2.expected_duration)


class TestTrialRunner(unittest.TestCase):

    def setUp(self):
        self._trial_result = mock.Mock()

        self._system = mock.create_autospec(core.system.VisionSystem)
        self._system.is_image_source_appropriate.return_value = True
        self._system.finish_trial.return_value = self._trial_result

        self._image_source = mock.create_autospec(core.image_source.ImageSource)
        self._image_source.get_stereo_baseline.return_value = None
        self._image_source._image_count = 0
        self._image_source._image = mock.Mock()

        def get_next_image(self_=self._image_source):
            self_._image_count += 1
            return self_._image, self_._image_count

        def is_complete(self_=self._image_source):
            return self_._image_count >= 10

        self._image_source.get_next_image.side_effect = get_next_image
        self._image_source.is_complete.side_effect = is_complete

    def test_run_system_checks_is_appropriate(self):
        self._system.is_image_source_appropriate.return_value = False
        task.run_system_with_source(self._system, self._image_source)
        self.assertIn(mock.call(self._image_source), self._system.is_image_source_appropriate.call_args_list)

    def test_run_system_calls_trial_functions_in_order(self):
        task.run_system_with_source(self._system, self._image_source)
        mock_calls = self._system.mock_calls
        # is_appropriate; set_camera_intrinsics; begin; 10 process image calls; end
        self.assertEqual(14, len(mock_calls))
        self.assertEqual('set_camera_intrinsics', mock_calls[1][0])
        self.assertEqual('start_trial', mock_calls[2][0])
        for i in range(10):
            self.assertEqual('process_image', mock_calls[3 + i][0])
        self.assertEqual('finish_trial', mock_calls[13][0])

    def test_run_system_calls_iteration_functions_in_order(self):
        task.run_system_with_source(self._system, self._image_source)
        mock_calls = self._image_source.mock_calls
        # get_camera_intrinsics; get_stereo_baseline; begin; 10 pairs of  is complete and get image; final is complete
        self.assertEqual(24, len(mock_calls))
        self.assertEqual('get_camera_intrinsics', mock_calls[0][0])
        self.assertEqual('get_stereo_baseline', mock_calls[1][0])
        self.assertEqual('begin', mock_calls[2][0])
        for i in range(10):
            self.assertEqual('is_complete', mock_calls[3 + 2 * i][0])
            self.assertEqual('get_next_image', mock_calls[4 + 2 * i][0])
        self.assertEqual('is_complete', mock_calls[23][0])

    def test_run_system_returns_trial_result(self):
        result = task.run_system_with_source(self._system, self._image_source)
        self.assertEqual(self._trial_result, result)
