import unittest
import unittest.mock as mock
import os.path
import numpy as np
import multiprocessing
import multiprocessing.queues
import bson.objectid as oid
import database.tests.test_entity
import util.dict_utils as du
import metadata.image_metadata as imeta
import core.sequence_type
import core.image
import systems.slam.orbslam2
import orbslam2


class TestORBSLAM2(database.tests.test_entity.EntityContract, unittest.TestCase):
    def get_class(self):
        return systems.slam.orbslam2.ORBSLAM2

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'vocabulary_file': 'imafile-{0}'.format(np.random.randint(10, 20)),
            'settings': {
                'Camera': {
                    'fx': np.random.uniform(10, 1000),
                    'fy': np.random.uniform(10, 1000),
                    'cx': np.random.uniform(10, 1000),
                    'cy': np.random.uniform(10, 1000),

                    'k1': np.random.uniform(-1, 1),
                    'k2': np.random.uniform(-1, 2),
                    'p1': np.random.uniform(0, 1),
                    'p2': np.random.uniform(0, 1),
                    'k3': np.random.uniform(-10, 10),
                    'fps': np.random.uniform(10, 100),
                    'RGB': np.random.randint(0, 1),
                },
                'ORBextractor': {
                    'nFeatures': np.random.randint(0, 8000),
                    'scaleFactor': np.random.uniform(0, 2),
                    'nLevels': np.random.randint(1, 10),
                    'iniThFAST': np.random.randint(0, 100),
                    'minThFAST': np.random.randint(0, 20)
                },
                'Viewer': {
                    'KeyFrameSize': np.random.uniform(0, 1),
                    'KeyFrameLineWidth': np.random.uniform(0, 3),
                    'GraphLineWidth': np.random.uniform(0, 3),
                    'PointSize': np.random.uniform(0, 3),
                    'CameraSize': np.random.uniform(0, 1),
                    'CameraLineWidth': np.random.uniform(0, 3),
                    'ViewpointX': np.random.uniform(0, 10),
                    'ViewpointY': np.random.uniform(0, 10),
                    'ViewpointZ': np.random.uniform(0, 10),
                    'ViewpointF': np.random.uniform(0, 10)
                }
            },
            'mode': systems.slam.orbslam2.SensorMode(np.random.randint(0, 3)),
            'resolution': tuple(np.random.randint(10, 1000, 2)),
            'temp_folder': 'imafolder-{0}'.format(np.random.randint(10, 20))
        })
        return systems.slam.orbslam2.ORBSLAM2(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param trial_result1:
        :param trial_result2:
        :return:
        """
        if (not isinstance(trial_result1, systems.slam.orbslam2.ORBSLAM2) or
                not isinstance(trial_result2, systems.slam.orbslam2.ORBSLAM2)):
            self.fail('object was not a ORBSLAM2')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1._vocabulary_file, trial_result2._vocabulary_file)
        self.assertEqual(trial_result1._orbslam_settings, trial_result2._orbslam_settings)
        self.assertEqual(trial_result1.get_settings(), trial_result2.get_settings())

    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_starts_a_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        subject = self.make_instance()
        with mock.patch('systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertEqual(systems.slam.orbslam2.run_orbslam, mock_multiprocessing.Process.call_args[1]['target'])
        self.assertTrue(mock_process.start.called)

    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_saves_settings_file(self, _):
        mock_open = mock.mock_open()
        temp_folder = 'thisisatempfolder'
        subject = self.make_instance(temp_folder=temp_folder)
        with mock.patch('systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_open.called)
        self.assertEqual(os.path.join(temp_folder, 'orb-slam2-settings-unregistered'), mock_open.call_args[0][0])

    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_uses_id_in_settings_file(self, _):
        mock_open = mock.mock_open()
        sys_id = oid.ObjectId()
        temp_folder = 'thisisatempfolder'
        subject = self.make_instance(temp_folder=temp_folder, id_=sys_id)
        with mock.patch('systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_open.called)
        self.assertEqual(os.path.join(temp_folder, 'orb-slam2-settings-{0}'.format(sys_id)), mock_open.call_args[0][0])

    @mock.patch('systems.slam.orbslam2.os.path.isfile', autospec=os.path.isfile)
    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_finds_available_file(self, _, mock_isfile):
        mock_open = mock.mock_open()
        temp_folder = 'thisisatempfolder'
        base_filename = os.path.join(temp_folder, 'orb-slam2-settings-unregistered')
        mock_isfile.side_effect = lambda x: x != base_filename + '-10'
        subject = self.make_instance(temp_folder=temp_folder)
        with mock.patch('systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_isfile.called)
        self.assertIn(mock.call(base_filename), mock_isfile.call_args_list)
        for idx in range(10):
            self.assertIn(mock.call(base_filename + '-{0}'.format(idx)), mock_isfile.call_args_list)
        self.assertTrue(mock_open.called)
        self.assertEqual(base_filename + '-10', mock_open.call_args[0][0])

    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_does_nothing_for_non_sequential_input(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_open = mock.mock_open()
        subject = self.make_instance()
        with mock.patch('systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
        self.assertFalse(mock_multiprocessing.Process.called)
        self.assertFalse(mock_process.start.called)
        self.assertFalse(mock_open.called)

    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_sends_image_and_depth_to_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_multiprocessing.Process.return_value = mock_process
        mock_multiprocessing.Queue.return_value = mock_queue

        mock_image_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        mock_depth_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        image = core.image.Image(data=mock_image_data, depth_data=mock_depth_data, metadata=imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            height=32, width=32, hash_=b'\x00\x00\x00\x00\x00\x00\x00\x01'
        ))

        subject = self.make_instance()
        with mock.patch('systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)

    @mock.patch('systems.slam.orbslam2.os', autospec=os)
    @mock.patch('systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_finish_trial_waits_for_output(self, mock_multiprocessing, mock_os):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_open = mock.mock_open()
        subject = self.make_instance()
        with mock.patch('systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
        # TODO: Finish testing finish_trial

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestRunOrbslam(unittest.TestCase):

    def test_calls_initialize_and_shutdown(self):
        mock_input_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_input_queue.get.side_effect = [None, None]
        mock_output_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_orbslam = mock.create_autospec(orbslam2)
        mock_system = mock.create_autospec(orbslam2.System)
        mock_orbslam.System.return_value = mock_system
        with mock.patch.dict('sys.modules', orbslam2=mock_orbslam):
            systems.slam.orbslam2.run_orbslam(mock_output_queue, mock_input_queue, '', '',
                                              systems.slam.orbslam2.SensorMode.RGBD, (1280, 720))
        self.assertTrue(mock_system.initialize.called)
        self.assertTrue(mock_system.shutdown.called)
