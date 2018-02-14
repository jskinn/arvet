# Copyright (c) 2017, John Skinner
import abc
import arvet.core.sequence_type
import arvet.core.image_source
import arvet.config.path_manager


class Simulator(arvet.core.image_source.ImageSource, metaclass=abc.ABCMeta):
    """
    A Simulator, which produces images interactively with the system under test.
    This is the base type for all image sources with an INTERACTIVE sequence type,
    and defines even more methods
    TODO: We will need more methods to control the simulator.
    TODO: Support multiple cameras
    """

    def resolve_paths(self, path_manager: arvet.config.path_manager.PathManager):
        """
        Most simulators will contain something stored on the hard drive,
        use the configured path manager to locate the actual simulator on this device.
        If the simulator can not be found, raise FileNotFoundError
        Do nothing if the simulator doesn't need any files.
        :param path_manager: The path manager to help us find the simulator
        :return: void
        """
        pass

    @property
    def sequence_type(self):
        """
        Get the sequence type.
        Simulators always produce images interactively
        :return:
        """
        return arvet.core.sequence_type.ImageSequenceType.INTERACTIVE

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

    @abc.abstractmethod
    def move_camera_to(self, pose):
        """
        Move the camera to a given pose, colliding with objects and stopping if blocked.
        :param pose: The destination pose
        :return:
        """
        pass

    @abc.abstractmethod
    def get_obstacle_avoidance_force(self, radius=1):
        """
        Get a force for obstacle avoidance.
        The simulator should get all objects within the given radius,
        and provide a net force away from all the objects, scaled by the distance to the objects.

        :param radius: Distance to detect objects, in meters
        :return: A repulsive force vector, as a numpy array
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
        :param fov: The field of view, in radians
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
