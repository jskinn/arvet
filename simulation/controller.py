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
    def update_state(self, delta_time, simulator):
        """
        Step the controller a time amount,
        and push changes to the simulator, such as the camera pose
        :return: void
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the controller state so we can restart the simulation
        :return: void
        """
        pass
