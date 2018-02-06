# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os
import re
import subprocess
import time

import numpy as np
import unrealcv

import arvet.core.image
import arvet.database.tests.test_entity
import arvet.metadata.image_metadata as imeta
import arvet.simulation.unrealcv.unrealcv_simulator as uecvsim
import arvet.util.dict_utils as du
import arvet.util.transform as tf
import arvet.util.unreal_transform as uetf
import arvet.util.image_utils as image_utils


# Patch the unrealcv Client API for mocking
class ClientPatch(unrealcv.Client):
    connect = unrealcv.BaseClient.connect
    disconnect = unrealcv.BaseClient.disconnect
    isconnected = unrealcv.BaseClient.isconnected


class TestUnrealCVSimulator(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return uecvsim.UnrealCVSimulator

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'executable_path': 'temp/test_project/test-{0}.sh'.format(np.random.randint(0, 1000)),
            'world_name': 'sim_world_{0}'.format(np.random.randint(0, 1000)),
            'environment_type': imeta.EnvironmentType(np.random.randint(0, 4)),
            'light_level': imeta.LightingLevel(np.random.randint(0, 6)),
            'time_of_day': imeta.TimeOfDay(np.random.randint(0, 6)),
            'config': {}
        })
        return uecvsim.UnrealCVSimulator(*args, **kwargs)

    def assert_models_equal(self, simulator1, simulator2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param simulator1:
        :param simulator2:
        :return:
        """
        if (not isinstance(simulator1, uecvsim.UnrealCVSimulator) or
                not isinstance(simulator2, uecvsim.UnrealCVSimulator)):
            self.fail('object was not a ORBSLAM2')
        self.assertEqual(simulator1.identifier, simulator2.identifier)
        self.assertEqual(simulator1._executable, simulator2._executable)
        self.assertEqual(simulator1._world_name, simulator2._world_name)
        self.assertEqual(simulator1._environment_type, simulator2._environment_type)
        self.assertEqual(simulator1._light_level, simulator2._light_level)
        self.assertEqual(simulator1._time_of_day, simulator2._time_of_day)
        # We're not comparing config, because we expect it to be set when the system is run

    def test_can_configure_stereo(self):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'stereo_offset': 1})
        self.assertTrue(subject.is_stereo_available)

        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'stereo_offset': 0})
        self.assertFalse(subject.is_stereo_available)

    def test_can_configure_depth(self):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'provide_depth': True})
        self.assertTrue(subject.is_depth_available)

        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'provide_depth': False})
        self.assertFalse(subject.is_depth_available)

    def test_can_configure_labels(self):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'provide_labels': True})
        self.assertTrue(subject.is_labels_available)

        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'provide_labels': False})
        self.assertFalse(subject.is_labels_available)

    def test_can_configure_world_normals(self):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world',
                                            config={'provide_world_normals': True})
        self.assertTrue(subject.is_normals_available)

        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world',
                                            config={'provide_world_normals': False})
        self.assertFalse(subject.is_normals_available)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_saves_settings(self,  *_):
        port = np.random.randint(0, 1000)
        width = np.random.randint(10, 1000)
        height = np.random.randint(10, 1000)
        subject = uecvsim.UnrealCVSimulator('temp/blah/notreal.sh', 'sim-world', config={
            'port': port,
            'resolution': {'width': width, 'height': height},
        })

        mock_path_manager = mock.Mock()
        mock_path_manager.find_file.return_value = 'temp/test_project/test.sh'
        subject.resolve_paths(mock_path_manager)

        mock_open = mock.mock_open()
        with mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock_open, create=True):
            subject.begin()
        self.assertTrue(mock_open.called)
        self.assertEqual(mock.call('temp/test_project/unrealcv.ini', 'w'), mock_open.call_args)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        file_contents = mock_file.write.call_args[0][0]
        self.assertIn('Port={0}'.format(port), file_contents)
        self.assertIn('Width={0}'.format(width), file_contents)
        self.assertIn('Height={0}'.format(height), file_contents)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    def test_begin_starts_simulator(self, mock_subprocess, *_):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/not-real.sh', 'sim-world')

        mock_path_manager = mock.Mock()
        mock_path_manager.find_file.return_value = 'temp/test_project/test.sh'
        subject.resolve_paths(mock_path_manager)

        subject.begin()
        self.assertTrue(mock_subprocess.Popen.called)
        self.assertEqual('temp/test_project/test.sh', mock_subprocess.Popen.call_args[0][0])

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_creates_client(self, mock_client, *_):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()
        self.assertTrue(mock_client.called)
        self.assertEqual(mock_client.call_args, mock.call(('localhost', 9000)))

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_connects_to_simulator(self, mock_client, *_):
        mock_client_instance = make_mock_unrealcv_client()
        mock_client_instance.isconnected.return_value = False
        mock_client.return_value = mock_client_instance

        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()
        self.assertTrue(mock_client_instance.connect.called)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_sets_camera_properties(self, mock_client, *_):
        mock_client_instance = make_mock_unrealcv_client()
        mock_client_instance.isconnected.return_value = True
        mock_client.return_value = mock_client_instance

        fov = np.random.uniform(1, 90)
        focus_distance = np.random.uniform(0, 10000)
        aperture = np.random.uniform(1, 24)
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={
            'fov': fov,
            'depth_of_field_enabled': True,
            'focus_distance': focus_distance,
            'aperture': aperture,
        })
        subject.begin()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call("vset /camera/0/fov {0}".format(fov)), mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /camera/0/fstop {0}".format(aperture)),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /camera/0/enable-dof 1"), mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /camera/0/autofocus 0"), mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /camera/0/focus-distance {0}".format(focus_distance * 100)),
                      mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_sets_quality_properties(self, mock_client, *_):
        mock_client_instance = make_mock_unrealcv_client()
        mock_client_instance.isconnected.return_value = True
        mock_client.return_value = mock_client_instance

        texture_mipmap_bias = np.random.randint(0, 10)
        normal_maps_enabled = np.random.randint(0, 2)
        roughness_enabled = np.random.randint(0, 2)
        geometry_decimation = np.random.randint(0, 15)
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={
            'texture_mipmap_bias': texture_mipmap_bias,
            'normal_maps_enabled': bool(normal_maps_enabled),
            'roughness_enabled': bool(roughness_enabled),
            'geometry_decimation': geometry_decimation,
        })
        subject.begin()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call("vset /quality/texture-mipmap-bias {0}".format(texture_mipmap_bias)),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /quality/normal-maps-enabled {0}".format(normal_maps_enabled)),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /quality/roughness-enabled {0}".format(roughness_enabled)),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call("vset /quality/geometry-decimation {0}".format(geometry_decimation)),
                      mock_client_instance.request.call_args_list)

    def test_get_next_image_returns_none_if_unstarted(self, *_):
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        result = subject.get_next_image()
        self.assertEqual(2, len(result))
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.os.remove', autospec=os.remove)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_lit(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()

        mock_client_instance.request.called = False
        subject.get_next_image()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call("vget /camera/0/lit"), mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.os.remove', autospec=os.remove)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_depth(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'provide_depth': True})
        subject.begin()

        mock_client_instance.request.called = False
        subject.get_next_image()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call("vget /camera/0/depth"), mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.os.remove', autospec=os.remove)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_labels(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={'provide_labels': True})
        subject.begin()

        mock_client_instance.request.called = False
        subject.get_next_image()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call("vget /camera/0/object_mask"), mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.os.remove', autospec=os.remove)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_world_normals(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world',
                                            config={'provide_world_normals': True})
        subject.begin()

        mock_client_instance.request.called = False
        subject.get_next_image()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call("vget /camera/0/normal"), mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.os.remove', autospec=os.remove)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_image_and_none(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client.return_value = make_mock_unrealcv_client()
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()
        result = subject.get_next_image()
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], arvet.core.image.Image)
        self.assertIsNone(result[1])

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.os.remove', autospec=os.remove)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_cleans_up_image_files(self, mock_client, mock_read_colour, mock_os_remove, *_):
        mock_read_colour.side_effect = (
            lambda x, _=None: make_mock_image('object_mask') if 'label' in x else make_mock_image(x))
        filenames = {
            'vget /camera/0/lit': '/tmp/001.png',
            'vget /camera/0/depth': '/tmp/001.depth.png',
            'vget /camera/0/object_mask': '/tmp/001.labels.png',
            'vget /camera/0/normal': '/tmp/001.normal.png'
        }
        mock_client_instance = make_mock_unrealcv_client()
        mock_client_instance.request.side_effect = (lambda x: filenames[x] if x in filenames
                                                    else mock_unrealcv_request(x))
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world', config={
            'provide_depth': True,
            'provide_labels': True,
            'provide_world_normals': True
        })
        subject.begin()
        subject.get_next_image()
        self.assertTrue(mock_os_remove.called)
        for path in filenames.values():
            self.assertIn(mock.call(path), mock_os_remove.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_camera_pose_handles_frame_conversion_to_unreal(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()

        pose = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))
        ue_pose = uetf.transform_to_unreal(pose)
        subject.set_camera_pose(pose)
        self.assertEqual(subject.current_pose, pose)
        self.assertIn(mock.call(("vset /camera/0/location {0} {1} {2}".format(ue_pose.location[0],
                                                                              ue_pose.location[1],
                                                                              ue_pose.location[2]))),
                      mock_client_instance.request.call_args_list)
        # We can't search by string, because the floats are slightly inconsistent
        match = None
        for call in mock_client_instance.request.call_args_list:
            match = re.match('vset /camera/0/rotation ([-0-9.]+) ([-0-9.]+) ([-0-9.]+)', call[0][0])
            if match is not None:
                break
        self.assertIsNotNone(match)
        self.assertAlmostEqual(ue_pose.pitch, float(match.group(1)))
        self.assertAlmostEqual(ue_pose.yaw, float(match.group(2)))
        self.assertAlmostEqual(ue_pose.roll, float(match.group(3)))

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_camera_pose_handles_frame_conversion_from_unreal(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()

        pose = tf.Transform((-175, 29, -870), (0.3, -0.2, -0.8, 0.6))
        ue_pose = uetf.transform_to_unreal(pose)
        subject.set_camera_pose(ue_pose)
        self.assertTrue(np.array_equal(subject.current_pose.location, pose.location))
        self.assertTrue(np.all(np.isclose(subject.current_pose.rotation_quat(True), pose.rotation_quat(True))))
        self.assertIn(mock.call(("vset /camera/0/location {0} {1} {2}".format(ue_pose.location[0],
                                                                              ue_pose.location[1],
                                                                              ue_pose.location[2]))),
                      mock_client_instance.request.call_args_list)

        # We can't search by string, because the floats are slightly inconsistent
        match = None
        for call in mock_client_instance.request.call_args_list:
            match = re.match('vset /camera/0/rotation ([-0-9.]+) ([-0-9.]+) ([-0-9.]+)', call[0][0])
            if match is not None:
                break
        self.assertIsNotNone(match)
        self.assertAlmostEqual(ue_pose.pitch, float(match.group(1)))
        self.assertAlmostEqual(ue_pose.yaw, float(match.group(2)))
        self.assertAlmostEqual(ue_pose.roll, float(match.group(3)))

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_field_of_view(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()

        subject.set_field_of_view(30.23)
        self.assertIn(mock.call("vset /camera/0/fov 30.23"), mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_focus_distance(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()

        subject.set_focus_distance(19.043)     # Distance is in meters
        self.assertIn(mock.call("vset /camera/0/focus-distance 1904.3"), mock_client_instance.request.call_args_list)

    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.open', mock.mock_open(), create=True)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.subprocess', autospec=subprocess)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.arvet.util.image_utils.read_colour',
                autospec=image_utils.read_colour)
    @mock.patch('arvet.simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_fstop(self, mock_client, mock_read_colour, *_):
        mock_read_colour.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator('temp/test_project/test.sh', 'sim-world')
        subject.begin()

        subject.set_fstop(22.23)
        self.assertIn(mock.call("vset /camera/0/fstop 22.23"), mock_client_instance.request.call_args_list)


def make_mock_unrealcv_client():
    mock_client = mock.create_autospec(ClientPatch)
    mock_client.request.side_effect = mock_unrealcv_request
    return mock_client


def mock_unrealcv_request(request):
    if re.match('vget /camera/\d*/(lit|depth|normal)', request):
        return 'imfile.png'
    elif re.match('vget /camera/\d*/object_mask', request):
        return 'object_mask.png'
    elif re.match('vget /camera/\d*/location', request):
        return "{0} {1} {2}".format(*tuple(np.random.uniform(-1000, 1000, 3)))    # Random point in space
    elif re.match('vget /camera/\d*/rotation', request):
        return "{0} {1} {2}".format(*tuple(np.random.uniform(-180, 180, 3)))      # Random roll, pitch, yaw
    elif re.match('vget /object/name', request):
        return 'labelled_object'    # Object name
    elif re.match('vget /[a-z_/0-9]+/labels', request):
        return 'class1,class2'
    elif re.match('vget', request):
        return str(np.random.uniform(0, 100))
    return None


def make_mock_image(filename, _=None):
    if re.match('object_mask', filename):
        mask = np.zeros((64, 64, 3), dtype='uint8')
        mask[12:34, 17:22] = np.random.randint(0, 255, 3)
        return mask
    return np.random.randint(0, 255, (64, 64, 3), dtype='uint8')
