import abc
import core.sequence_type
import core.image_source


class Simulator(core.image_source.ImageSource, metaclass=abc.ABCMeta):
    """
    A Simulator, which produces images interactively with the system under test.
    This is the base type for all image sources with an INTERACTIVE sequence type,
    and defines even more methods
    TODO: We will need more methods to control the simulator.
    TODO: Support multiple cameras
    """

    @property
    def sequence_type(self):
        """
        Get the sequence type.
        Simulators always produce images interactively
        :return:
        """
        return core.sequence_type.ImageSequenceType.INTERACTIVE

    @property
    @abc.abstractmethod
    def current_pose(self):
        """
        Get the current pose of the camera in the simulator
        :return: A pose object that is the current location of the camera within the simulator
        """
        pass

    @abc.abstractmethod
    def set_camera_pose(self, pose):
        """
        Set the camera pose for the simulator
        :param pose:
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def field_of_view(self):
        """
        Get the current FOV of the simulator
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

    @property
    @abc.abstractmethod
    def focus_distance(self):
        """
        Get the current focus distance of the camera in the simulator
        :return:
        :rtype float:
        """
        pass

    @abc.abstractmethod
    def set_focus_distance(self, focus_distance):
        """
        If possible, set the focus distance of the simulated camera
        :param focus_distance:
        :return:
        :rtype float:
        """
        pass

    @property
    @abc.abstractmethod
    def fstop(self):
        """
        Get the current camera aperture setting
        :return:
        :rtype float:
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

    @abc.abstractmethod
    def shutdown(self):
        """
        Shut down the simulator.
        At the moment, this is less relevant for other image source types.
        If it becomes more common, move it into image_source
        :return:
        """
        pass

    @abc.abstractmethod
    def get_camera_matrix(self):
        """
        Get the current camera matrix of the current camera
        :return:
        """

    @property
    def supports_random_access(self):
        """
        We can't normally randomly access from simulators, since it needs input.
        If a particular simulator is different, override this
        :return:
        """
        return False

    def get(self, index):
        """
        Random access is not supported for simulators in general,
        specific simulators may override this
        :param index:
        :return:
        """
        return None
