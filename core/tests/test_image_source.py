#Copyright (c) 2017, John Skinner
import abc


class ImageSourceContract(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def make_instance(self, *args, **kwargs):
        """
        Make a new instance of the entity with default arguments.
        Parameters passed to this function should override the defaults.
        :param args: Forwarded to the constructor
        :param kwargs: Forwarded to the constructor
        :return: A new instance of the class under test
        """
        return None

    def test_iteration(self):
        """
        hmmmm....
        :return:
        """
        subject = self.make_instance()

        subject.begin()
        while not subject.is_complete():
            image, index = subject.get_next_image()
        self.assertEqual((None, None), subject.get_next_image())

    def test_begin_restarts(self):
        subject = self.make_instance()

        subject.begin()
        while not subject.is_complete():
            image, index = subject.get_next_image()
