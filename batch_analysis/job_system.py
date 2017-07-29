import abc


class JobSystem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def can_generate_dataset(self, simulator, config):
        """
        Can this job system generate synthetic datasets.
        This requires more setup than other jobs, so we have this check.
        Checks for other jobs would make sense, we jut haven't bothered yet.
        :param simulator: The simulator id that will be doing the generation
        :param config: Configuration passed to the simulator at run time
        :return: True iff the job system can generate datasets. HPC cannot.
        """
        pass

    @abc.abstractmethod
    def queue_generate_dataset(self, simulator_id, config, experiment=None):
        """
        Queue generating a synthetic dataset using a particular simulator
        and a particular configuration
        :param simulator_id: The id of the simulator to use to generate the dataset
        :param config: Configuration passed to the simulator to control the dataset generation
        :param experiment: The experiment this generated dataset is associated with, if any
        :return: True iff the job was successfully queued
        """
        pass

    @abc.abstractmethod
    def queue_train_system(self, trainer_id, trainee_id, experiment=None):
        """
        Use the job system to make a trainer train a particular trainee to produce a new system

        This should not run the job immediately, it may depend on some state
        that has not been saved yet, defer the execution of jobs until we've finished creating them.

        :param trainer_id: The id of the trainer to do the training
        :param trainee_id: The id of the trainee to train
        :param experiment: The experiment associated with this run, if any, to attach the new system to
        :return: True iff the job was queued
        """
        pass

    @abc.abstractmethod
    def queue_run_system(self, system_id, image_source_id, experiment=None):
        """
        Use the job system to run a system with a particular image source.

        This should not run the job immediately, it may depend on some state
        that has not been saved yet, defer the execution of jobs until we've finished creating them.

        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param experiment: The experiment associated with this run, if any
        :return: True iff the job was successfully queued
        """
        pass

    @abc.abstractmethod
    def queue_benchmark_result(self, trial_id, benchmark_id, experiment=None):
        """
        Use the job system to benchmark a particular trial result.
        Do not actually start the job yet, it may depend on

        :param trial_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param experiment: The experiment this is associated with, if any
        :return: True iff the job was successfully queued
        """
        pass

    @abc.abstractmethod
    def push_queued_jobs(self):
        """
        Everything is ready, actually start the jobs.
        This kind of deferred job queueing is done
        so that we can create jobs and update state that they depend on
        together, and the jobs will still be run with the changed state
        :return:
        """
        pass
