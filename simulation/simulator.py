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
