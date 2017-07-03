import unittest
import unittest.mock as mock

import numpy as np
import cv2
import unrealcv

import core.image
import simulation.controller
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
        subject = uecvsim.UnrealCVSimulator(None, config)
        self.assertTrue(subject.is_depth_available)

        config = {'provide_depth': False}
        subject = uecvsim.UnrealCVSimulator(None, config)
        self.assertFalse(subject.is_depth_available)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_begin_creates_client(self, mock_client, _):
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertTrue(mock_client.called)
        self.assertEqual(mock_client.call_args, mock.call(('localhost', 9000)))

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_begin_connects_to_server(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client_instance.isconnected.return_value = False
        mock_client.return_value = mock_client_instance

        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertTrue(mock_client_instance.connect.called)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_begin_resets_controller(self, *_):
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertTrue(controller.reset.called)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_none_if_unstarted(self, *_):
        subject = uecvsim.UnrealCVSimulator(None, None)
        result = subject.get_next_image()
        self.assertEqual(2, len(result))
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_updates_state_from_controller(self, *_):
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertFalse(controller.update_state.called)
        subject.get_next_image()
        self.assertTrue(controller.update_state.called)
        self.assertIn(mock.call(1/30, subject), controller.update_state.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_lit(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/lit"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_depth(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, {'provide_depth': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/depth"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_labels(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, {'provide_labels': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/object_mask"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_world_normals(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, {'provide_world_normals': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/normal"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_image_and_timestamp(self, *_):
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        result = subject.get_next_image()
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], core.image.Image)
        self.assertEqual(1/30, result[1])

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_set_camera_pose_handles_frame_conversion(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
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
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_set_field_of_view(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()

        subject.set_field_of_view(30.23)
        self.assertIn(mock.call("vset /camera/0/fov 30.23"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_set_focus_distance(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()

        subject.set_focus_distance(190.43)
        self.assertIn(mock.call("vset /camera/0/focus-distance 190.43"), mock_client_instance.request.call_args_list)

    @mock.patch('simulation.unrealcv.unrealcv_simulator.cv2', autospec=cv2)
    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_set_fstop(self, mock_client, _):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()

        subject.set_fstop(22.23)
        self.assertIn(mock.call("vset /camera/0/fstop 22.23"), mock_client_instance.request.call_args_list)
