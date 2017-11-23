# Copyright (c) 2017, John Skinner
import abc


class Identifiable(metaclass=abc.ABCMeta):
    """
    A superclass for classes that can be referred to in the database.
    They have an identifier.
    All entities are identifiable, although other things may also be identifiable.
    In particular, some classes have static identifiers and are referred to in the database.
    """

    @property
    @abc.abstractmethod
    def identifier(self):
        """
        Get the ID of the class, to which things will be associated in the database.
        This should be consistent for all instances of a particular class,
        It would be a class constant if I knew a way of making that abstract.
        :return: A string key to identify the particular class.
        :rtype: str
        """
        pass