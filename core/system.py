import abc
import database.referenced_class


class VisionSystem(database.referenced_class.ReferencedClass, metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def type(self):
        """
        Get the type of the system
        :return:
        :rtype: VisionSystemType
        """
        pass

    @abc.abstractproperty
    def is_deterministic(self):
        """
        Is the visual system deterministic.

        If this is false, it will have to be tested multiple times, because the performance will be inconsistent
        between runs.

        :return: True iff the algorithm will produce the same results each time.
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def get_dataset_criteria(self):
        """
        Get search criteria for finding datasets that can be processed by this system.
        I'm not sure what the best way to do this is yet.
        :return: A set of criteria for finding datasets that match
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def is_dataset_appropriate(self, dataset):
        """
        Is the dataset appropriate for testing this vision system.
        :param dataset: A dataset object.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def run_with_dataset(self, dataset_images):
        """
        Run the system over a given dataset, to produce a trial result.
        :param dataset_images: A set of images to run over
        :return: The trial results
        :rtype: TrialResult
        """
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is VisionSystem:
            for s in subclass.__mro__:
                if all(name in s.__dict__ for name in dir(VisionSystem)):
                    return True
        return NotImplemented
