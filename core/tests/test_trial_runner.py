import unittest, unittest.mock
import core.trial_runner as runner
import core.system
import core.image_source


class MockSystem(core.system.VisionSystem):
    pass


class MockImageSource(core.image_source.ImageSource):
    pass


class TestTrialRunner(unittest.TestCase):

    def setUp(self):
        self._trial_result = unittest.mock.Mock()

        self._system = unittest.mock.Mock(spec=core.system.VisionSystem)
        self._system.is_image_source_appropriate.return_value = True
        self._system.finish_trial.return_value = self._trial_result

        self._image_source = unittest.mock.Mock(spec=core.image_source.ImageSource)
        self._image_source._image_count = 0
        self._image_source._image = unittest.mock.Mock()
        def get_next_image(self=self._image_source):
            self._image_count += 1
            return self._image
        def is_complete(self=self._image_source):
            return self._image_count >= 10
        self._image_source.get_next_image.side_effect = get_next_image
        self._image_source.is_complete.side_effect = is_complete

    def test_run_system_checks_is_appropriate(self):
        self._system.is_image_source_appropriate.return_value = False
        runner.run_system_with_source(self._system, self._image_source)
        self._system.is_image_source_appropriate.assert_called_once_with(self._image_source)

    def test_run_system_calls_trial_functions_in_order(self):
        runner.run_system_with_source(self._system, self._image_source)
        mock_calls = self._system.mock_calls
        self.assertEquals(13, len(mock_calls)) # is_appropriate; begin; 10 process image calls; end
        self.assertEquals('start_trial', mock_calls[1][0])
        for i in range(10):
            self.assertEquals('process_image', mock_calls[2 + i][0])
        self.assertEquals('finish_trial', mock_calls[12][0])

    def test_run_system_calls_iteration_functions_in_order(self):
        runner.run_system_with_source(self._system, self._image_source)
        mock_calls = self._image_source.mock_calls
        self.assertEquals(22, len(mock_calls))  # begin; 10 pairs of  is complete and get image; final is complete
        self.assertEquals('begin', mock_calls[0][0])
        for i in range(10):
            self.assertEquals('is_complete', mock_calls[1 + 2 * i][0])
            self.assertEquals('get_next_image', mock_calls[2 + 2 * i][0])
        self.assertEquals('is_complete', mock_calls[21][0])

    def test_run_system_returns_trial_result(self):
        result = runner.run_system_with_source(self._system, self._image_source)
        self.assertEquals(self._trial_result, result)
