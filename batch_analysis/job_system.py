import abc


class JobSystem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run_script(self, script_name, *args):
        """
        Run a given script, with command line arguments.
        :param script_name: The name of the script to run, as a string.
        :param args: Additional arguments passed to the script, space delimited. Must all be strings.
        :return: 
        """
        pass

