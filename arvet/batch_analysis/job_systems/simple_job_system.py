# Copyright (c) 2017, John Skinner
import logging
import typing
import arvet.batch_analysis.job_system
from arvet.batch_analysis.task import Task
import arvet.batch_analysis.scripts.run_task


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
        super().__init__(config)
        self._queue = []

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

    def run_task(self, task: Task) -> typing.Union[int, None]:
        """
        Run a particular task
        :param task: The task object to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        if self.can_run_task(task):
            return self.run_script(
                script=arvet.batch_analysis.scripts.run_task.__file__,
                script_args=[str(task.identifier)],
                num_cpus=task.num_cpus,
                num_gpus=task.num_gpus,
                memory_requirements=task.memory_requirements,
                expected_duration=task.expected_duration
            )
        return None

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
        self._queue.append((script, script_args))
        return len(self._queue) - 1  # Job id is the index in the queue

    def run_queued_jobs(self):
        """
        Actually run the jobs.
        Does so in a subprocess to maintain a clean state between tasks.
        :return: void
        """
        import subprocess
        logging.getLogger(__name__).info("Running {0} jobs...".format(len(self._queue)))
        for script_path, script_args in self._queue:
            subprocess.run(['python', script_path] + script_args)
        self._queue = []
