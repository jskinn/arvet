# Copyright (c) 2017, John Skinner
import logging
import typing
import functools
import bson
import arvet.batch_analysis.job_system
import arvet.run_task


class SimpleJobSystem(arvet.batch_analysis.job_system.JobSystem):
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

    def run_task(self, task_id, num_cpus: int = 1, num_gpus: int = 0,
                 memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
        """
        Run a particular task
        :param task_id: The id of the task to run
        :param num_cpus: The number of CPUs required. Ignored.
        :param num_gpus: The number of GPUs required. Ignored.
        :param memory_requirements: The required amount of memory. Ignored.
        :param expected_duration: The duration given for the job to run. Ignored.
        :return: The job id if the job has been started correctly, None if failed.
        """
        self._queue.append(functools.partial(run_task, task_id))
        return len(self._queue) - 1     # Job id is the index in the queue

    def run_script(self, script: str, script_args: typing.List[str], num_cpus: int = 1, num_gpus: int = 0,
                   memory_requirements: str = '3GB', expected_duration: str = '1:00:00') -> int:
        """
        Run a script that is not a task on this job system
        :param script: The path to the script to run
        :param script_args: A list of command line arguments, as strings
        :param num_cpus: The number of CPUs required
        :param num_gpus: The number of GPUs required
        :param memory_requirements: The required amount of memory
        :param expected_duration: The duration given for the job to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        self._queue.append(functools.partial(run_script, script, script_args))
        return len(self._queue) - 1  # Job id is the index in the queue

    def run_queued_jobs(self):
        """
        Actually run the jobs.
        :return: void
        """
        logging.getLogger(__name__).info("Running {0} jobs...".format(len(self._queue)))
        for partial in self._queue:
            partial()
        self._queue = []


def run_task(task_id: bson.ObjectId):
    """
    Tiny helper to invoke running a task
    :param task_id:
    :return:
    """
    arvet.run_task.main(str(task_id))


def run_script(script_path: str, script_args: typing.List[str]):
    """
    Helper that invokes a script
    :param script_path:
    :param script_args:
    :return:
    """
    import subprocess
    subprocess.run(['python', script_path] + script_args)
