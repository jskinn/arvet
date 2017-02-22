import abc


class ReferencedClass(metaclass=abc.ABCMeta):
    """
    A superclass for classes that can be referred to in the database.
    They have an identifier.
    """

    @abc.abstractproperty
    def identifier(self):
        """
        Get the ID of the class, to which things will be associated in the database.
        This should be consistent for all instances of a particular class,
        It would be a class constant if I knew a way of making that abstract.
        :return: A string key to identify the particular class.
        :rtype: str
        """
        pass