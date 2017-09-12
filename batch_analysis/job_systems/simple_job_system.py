#Copyright (c) 2017, John Skinner
import logging
import batch_analysis.job_system
import run_task


class SimpleJobSystem(batch_analysis.job_system.JobSystem):
    """
    The worst possible, and simplest, job system.
    Simply does the job as part of scheduling it.
    No multiprocess, nothing, just direct execution.
    Still implements a job queueing system,
    so that we can defer the execution of jobs until we've finished creating them.
    It does ignore provided job requirements.
    """

    def __init__(self, config):
        self._node_id = config['node_id'] if 'node_id' in config else 'simple-job-system'
        self._queue = []

    @property
    def node_id(self):
        """
        All job systems should have a node id, controlled by the configuration.
        The idea is that different job systems on different computers have different
        node ids, so that we can track which system is supposed to be running which job id.
        :return:
        """
        return self._node_id

    def can_generate_dataset(self, simulator, config):
        """
        Can this job system generate synthetic datasets?
        This job system can.
        :param simulator: The simulator id that will be doing the generation
        :param config: Configuration passed to the simulator at run time
        :return: True
        """
        return True

    def is_job_running(self, job_id):
        """
        Is the specified job id currently running through this job system.
        This is used by the task manager to work out which jobs have failed without notification, to reschedule them.
        For the simple job system, a job is "running" if it is a valid index in the queue.
        :param job_id: The integer job id to check
        :return: True if the job is currently running on this node
        """
        return 0 <= job_id < len(self._queue)

    def run_task(self, task_id, num_cpus=1, num_gpus=0, memory_requirements='3GB',
                 expected_duration='1:00:00'):
        """
        Run a particular task
        :param task_id: The id of the task to run
        :param num_cpus: The number of CPUs required. Ignored.
        :param num_gpus: The number of GPUs required. Ignored.
        :param memory_requirements: The required amount of memory. Ignored.
        :param expected_duration: The duration given for the job to run. Ignored.
        :return: The job id if the job has been started correctly, None if failed.
        """
        self._queue.append(str(task_id))
        return len(self._queue) - 1     # Job id is the index in the queue

    def run_queued_jobs(self):
        """
        Actually run the jobs.
        :return: void
        """
        logging.getLogger(__name__).info("Running {0} jobs...".format(len(self._queue)))
        for task_id in self._queue:
            run_task.main(task_id)
        self._queue = []
