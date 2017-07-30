import unittest
import unittest.mock as mock
import os
import re

import numpy as np
import cv2
import unrealcv

import core.image
import simulation.unrealcv.unrealcv_simulator as uecvsim
import util.transform as tf
import util.unreal_transform as uetf


# Patch the unrealcv Client API for mocking
class ClientPatch(unrealcv.Client):
    connect = unrealcv.BaseClient.connect
    disconnect = unrealcv.BaseClient.disconnect
    isconnected = unrealcv.BaseClient.isconnected


# TODO: Maybe create a mixin for ImageSource
class TestUnrealCVSimulator(unittest.TestCase):

    def test_can_configure_depth(self):
        config = {'provide_depth': True}
        subject = uecvsim.UnrealCVSimulator(config)
        self.assertTrue(subject.is_depth_available)

        config = {'provide_depth': False}
        subject = uecvsim.UnrealCVSimulator(config)
        self.assertFalse(subject.is_depth_available)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_creates_client(self, mock_client, mock_cv2):
        mock_cv2.imread.side_effect = make_mock_image
        subject = uecvsim.UnrealCVSimulator()
        subject.begin()
        self.assertTrue(mock_client.called)
        self.assertEqual(mock_client.call_args, mock.call(('localhost', 9000)))

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_begin_connects_to_server(self, mock_client, mock_cv2):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client_instance.isconnected.return_value = False
        mock_client.return_value = mock_client_instance

        subject = uecvsim.UnrealCVSimulator()
        subject.begin()
        self.assertTrue(mock_client_instance.connect.called)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_none_if_unstarted(self, *_):
        subject = uecvsim.UnrealCVSimulator()
        result = subject.get_next_image()
        self.assertEqual(2, len(result))
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_lit(self, mock_client, mock_cv2, _):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator()
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/lit"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_depth(self, mock_client, mock_cv2, _):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator({'provide_depth': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/depth"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_labels(self, mock_client, mock_cv2, _):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator({'provide_labels': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/object_mask"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_world_normals(self, mock_client, mock_cv2, _):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator({'provide_world_normals': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/normal"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_image_and_none(self, mock_client, mock_cv2, _):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client.return_value = make_mock_unrealcv_client()
        subject = uecvsim.UnrealCVSimulator()
        subject.begin()
        result = subject.get_next_image()
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], core.image.Image)
        self.assertIsNone(result[1])

    @mock.patch('simulation.unrealcv.unrealcv_simulator.os', autospec=os)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_cleans_up_image_files(self, mock_client, mock_cv2, mock_os):
        mock_cv2.imread.side_effect = lambda x: make_mock_image('object_mask') if 'label' in x else make_mock_image(x)
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
        subject = uecvsim.UnrealCVSimulator({
            'provide_depth': True,
            'provide_labels': True,
            'provide_world_normals': True
        })
        subject.begin()
        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        for path in filenames.values():
            self.assertIn(mock.call(path), mock_os.remove.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_camera_pose_handles_frame_conversion(self, mock_client, mock_cv2):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator()
        subject.begin()

        pose = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))
        ue_pose = uetf.transform_to_unreal(pose)
        subject.set_camera_pose(pose)
        self.assertEqual(subject.current_pose, pose)
        self.assertIn(mock.call(("vset /camera/0/location {0} {1} {2}".format(ue_pose.location[0],
                                                                              ue_pose.location[1],
                                                                              ue_pose.location[2]))),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call(("vset /camera/0/rotation {0} {1} {2}".format(ue_pose.pitch,
                                                                              ue_pose.yaw,
                                                                              ue_pose.roll))),
                      mock_client_instance.request.call_args_list)

        pose = tf.Transform((-175, 29, -870), (0.3, -0.2, -0.8, 0.6))
        ue_pose = uetf.transform_to_unreal(pose)
        subject.set_camera_pose(ue_pose)
        self.assertTrue(np.array_equal(subject.current_pose.location, pose.location))
        self.assertTrue(np.all(np.isclose(subject.current_pose.rotation_quat(True), pose.rotation_quat(True))))
        self.assertIn(mock.call(("vset /camera/0/location {0} {1} {2}".format(ue_pose.location[0],
                                                                              ue_pose.location[1],
                                                                              ue_pose.location[2]))),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call(("vset /camera/0/rotation {0} {1} {2}".format(ue_pose.pitch,
                                                                              ue_pose.yaw,
                                                                              ue_pose.roll))),
                      mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_field_of_view(self, mock_client, mock_cv2):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator()
        subject.begin()

        subject.set_field_of_view(30.23)
        self.assertIn(mock.call("vset /camera/0/fov 30.23"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_focus_distance(self, mock_client, mock_cv2):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator()
        subject.begin()

        subject.set_focus_distance(190.43)
        self.assertIn(mock.call("vset /camera/0/focus-distance 190.43"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('simulation.unrealcv.unrealcv_simulator.unrealcv.Client', autospec=ClientPatch)
    def test_set_fstop(self, mock_client, mock_cv2):
        mock_cv2.imread.side_effect = make_mock_image
        mock_client_instance = make_mock_unrealcv_client()
        mock_client.return_value = mock_client_instance
        subject = uecvsim.UnrealCVSimulator()
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


def make_mock_image(filename):
    if re.match('object_mask', filename):
        mask = np.zeros((64, 64, 3), dtype='uint8')
        mask[12:34, 17:22] = np.random.randint(0, 255, 3)
        return mask
    return np.random.randint(0, 255, (64, 64, 3), dtype='uint8')
