# Copyright (c) 2017, John Skinner
import abc
import argus.core.image_source


class Controller(argus.core.image_source.ImageSource, metaclass=abc.ABCMeta):
    """
    A controller wraps an interactive image source and turns it into a static one.
    That is, it controls the simulator in a predefined way to output images without requiring input.
    It changes an INTERACTIVE sequence type to either SEQUENTIAL or NON_SEQUENTIAL
    """

    @abc.abstractmethod
    def can_control_simulator(self, simulator):
        """
        Can this controller
        :param simulator:
        :return:
        """
        pass

    @abc.abstractmethod
    def set_simulator(self, simulator):
        """
        Set the simulator being controlled by this controller
        :param simulator:
        :return:
        """
        pass
