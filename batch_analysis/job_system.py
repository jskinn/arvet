import abc


class JobSystem(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def node_id(self):
        """
        All job systems should have a node id, controlled by the configuration.
        The idea is that different job systems on different computers have different
        node ids, so that we can track which system is supposed to be running which job id.
        :return:
        """
        pass

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
    def is_job_running(self, job_id):
        """
        Is the specified job id currently running through this job system.
        This is used by the task manager to work out which jobs have failed without notification, to reschedule them
        :param job_id: The integer job id to check
        :return: True if the job is currently running on this node
        """
        return False

    @abc.abstractmethod
    def run_task(self, task_id, num_cpus=1, num_gpus=0, memory_requirements='3GB',
                 expected_duration='1:00:00'):
        """
        Run a particular task
        :param task_id: The id of the task to run
        :param num_cpus: The number of CPUs required
        :param num_gpus: The number of GPUs required
        :param memory_requirements: The required amount of memory
        :param expected_duration: The duration given for the job to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        pass

    @abc.abstractmethod
    def run_queued_jobs(self):
        """
        Everything is ready, actually start the jobs.
        This kind of deferred job queueing is done
        so that we can create jobs and update state that they depend on
        together, and the jobs will still be run with the changed state
        :return:
        """
        pass
