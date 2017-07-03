import abc
import core.image_source


class Simulator(core.image_source.ImageSource, metaclass=abc.ABCMeta):
    """
    A Simulator, which can produce images on demand.
    But it needs a controller, which tells the camera how to move.
    Some systems are also controllers, and some systems require default controllers.
    Different controllers produce different image sequences.
    """

    def __init__(self, controller=None, **kwargs):
        super().__init__(**kwargs)
        self._controller = controller

    @property
    def controller(self):
        return self._controller

    @property
    def sequence_type(self):
        return self._controller.motion_type

    @property
    @abc.abstractmethod
    def current_pose(self):
        """
        Get the current pose of the camera in the simulator
        :return: A pose object that is the current location of the camera within the simulator
        """
        pass

    def set_controller(self, controller):
        """
        Set the controller used by this simulator.
        This may be overridden to respond to the controller in some way.
        :param controller:
        :return:
        """
        self._controller = controller

    @abc.abstractmethod
    def set_camera_pose(self, pose):
        """
        Set the camera pose for the simulator
        :param pose:
        :return:
        """
        pass

    @abc.abstractmethod
    def set_field_of_view(self, fov):
        """
        If possible, set the field of view of the simulated camera
        :param fov:
        :return:
        """
        pass

    @abc.abstractmethod
    def set_focus_distance(self, focus_distance):
        """
        If possible, set the focus distance of the simulated camera
        :param focus_distance:
        :return:
        """
        pass

    @abc.abstractmethod
    def set_fstop(self, fstop):
        """
        IF possible, set the aperture fstop of the simulated camera
        :param fstop:
        :return:
        """
        pass
