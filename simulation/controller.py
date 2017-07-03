import abc


class Controller(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def motion_type(self):
        """
        Get the kind of image sequence produced by this controller in a simulator.
        :return:
        """
        pass

    @abc.abstractmethod
    def is_complete(self):
        """
        Is the motion for this controller complete.
        Some controllers will produce finite motion, some will not.
        Those that do not simply always return false here.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_next_pose(self):
        """
        Get the next camera pose to use.
        This needs to be a 6-dof transform, look at using a library like transformation3d
        TODO: Need a way of providing more input to the simulator, beyond just the camera pose.
        :return:
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the controller state so we can restart the simulation
        :return: void
        """
        pass
