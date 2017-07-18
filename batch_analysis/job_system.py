import abc


class JobSystem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def schedule_run_system(self, system_id, image_source_id, experiment=None):
        """
        Use the job system to run a system with a particular image source.
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param experiment: The experiment associated with this run, if any
        :return: void
        """
        pass

    @abc.abstractmethod
    def schedule_benchmark_result(self, trial_id, benchmark_id, experiment=None):
        """
        Use the job system to benchmark a particular trial result.
        :param trial_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param experiment: The experiment this is associated with, if any
        :return: void
        """
        pass
