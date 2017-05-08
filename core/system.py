import abc
import database.entity


class VisionSystem(database.entity.Entity, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A Vision system, something that will be run, benchmarked, and analysed by this program.
    This is the standard interface that everything must implement to work with this system.
    All systems must be entities and stored in the database, so that the framework can load them, and 
    """

    @property
    @abc.abstractmethod
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
    def is_image_source_appropriate(self, image_source):
        """
        Is the dataset appropriate for testing this vision system.
        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def start_trial(self):
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :return: void
        """
        pass

    @abc.abstractmethod
    def process_image(self, image):
        """
        Process an image as part of the current run.
        Should automatically start a new trial if none is currently started.
        :param image:
        :return: void
        """
        pass

    @abc.abstractmethod
    def finish_trial(self):
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        return None
