import unittest
import unittest.mock as mock
import unrealcv
import util.transform as tf
import core.image
import simulation.controller
import simulation.unrealcv.unrealcv_simulator as uecvsim
import simulation.unrealcv.unreal_transform as uetf


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

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_begin_creates_client(self, mock_client):
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertTrue(mock_client.called)
        self.assertEquals(mock_client.call_args, mock.call(('localhost', 9000)))

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_begin_connects_to_server(self, mock_client):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client_instance.isconnected.return_value = False
        mock_client.return_value = mock_client_instance

        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertTrue(mock_client_instance.connect.called)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_begin_resets_controller(self, mock_client):
        controller = mock.Mock(spec=simulation.controller.Controller)
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertTrue(controller.reset.called)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_none_if_unstarted(self, mock_client):
        subject = uecvsim.UnrealCVSimulator(None, None)
        self.assertIsNone(subject.get_next_image())

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_gets_pose_from_controller(self, mock_client):
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        self.assertFalse(controller.get_next_pose.called)
        subject.get_next_image()
        self.assertTrue(controller.get_next_pose.called)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_sets_camera_pose(self, mock_client):
        pose = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))
        unreal_pose = uetf.transform_to_unreal(pose)

        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = pose
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertTrue(mock_client_instance.request.called)
        self.assertIn(mock.call(("vset /camera/0/location {0} {1} {2}".format(unreal_pose.location[0],
                                                                              unreal_pose.location[1],
                                                                              unreal_pose.location[2]))),
                      mock_client_instance.request.call_args_list)
        self.assertIn(mock.call(("vset /camera/0/rotation {0} {1} {2}".format(unreal_pose.pitch,
                                                                              unreal_pose.yaw,
                                                                              unreal_pose.roll))),
                      mock_client_instance.request.call_args_list)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_lit(self, mock_client):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))

        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/lit"), mock_client_instance.request.call_args_list)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_depth(self, mock_client):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))

        subject = uecvsim.UnrealCVSimulator(controller, {'provide_depth': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/depth"), mock_client_instance.request.call_args_list)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_labels(self, mock_client):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))

        subject = uecvsim.UnrealCVSimulator(controller, {'provide_labels': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/object_mask"), mock_client_instance.request.call_args_list)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_captures_world_normals(self, mock_client):
        mock_client_instance = mock.create_autospec(ClientPatch)
        mock_client.return_value = mock_client_instance
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))

        subject = uecvsim.UnrealCVSimulator(controller, {'provide_world_normals': True})
        subject.begin()

        self.assertFalse(mock_client_instance.request.called)
        subject.get_next_image()
        self.assertIn(mock.call("vget /camera/0/normal"), mock_client_instance.request.call_args_list)

    @mock.patch('unrealcv.Client', autospec=ClientPatch)
    def test_get_next_image_returns_images(self, mock_client):
        controller = mock.Mock(spec=simulation.controller.Controller)
        controller.get_next_pose.return_value = tf.Transform((17, -21, 3), (0.1, 0.7, -0.3, 0.5))
        subject = uecvsim.UnrealCVSimulator(controller, None)
        subject.begin()
        image = subject.get_next_image()
        self.assertIsInstance(image, core.image.Image)
