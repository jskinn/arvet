# Copyright (c) 2017, John Skinner
import logging
import typing
import os
import re
import subprocess
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

    def __init__(self, config, config_file):
        super().__init__(config)
        self._use_subprocess = bool(config.get('subprocess', False))
        self._queue = []
        self._config_path = os.path.abspath(config_file)

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

    def run_task(self, task: Task) -> bool:
        """
        Run a particular task
        :param task: The task object to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        if self.can_run_task(task):
            if self._use_subprocess:
                def args_builder(config):
                    return ['--config', config, str(task.pk)]
                job_id = self.run_script(
                    script=arvet.batch_analysis.scripts.run_task.__file__,
                    script_args_builder=args_builder,
                    num_cpus=task.num_cpus,
                    num_gpus=task.num_gpus,
                    memory_requirements=task.memory_requirements,
                    expected_duration=task.expected_duration
                )
                task.mark_job_started(self.node_id, job_id)
                task.save()
            else:
                # Add just the task id to the queue, that's all we actually do here
                self._queue.append((str(task.pk), None))
                task.mark_job_started(self.node_id, len(self._queue) - 1)  # Job id is the index in the queue
                task.save()
            return True
        return False

    def run_script(
            self,
            script: str,
            script_args_builder: typing.Callable[..., typing.List[str]],
            job_name: str = "",
            num_cpus: int = 1, num_gpus: int = 0,
            memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
    ) -> int:
        """
        Run a script that is not a task on this job system
        :param script: The path to the script to run
        :param script_args_builder: A function that returns a list of command line arguments, as strings
        :param job_name: A unique name for the job
        :param num_cpus: The number of CPUs required
        :param num_gpus: The number of GPUs required
        :param memory_requirements: The required amount of memory
        :param expected_duration: The duration given for the job to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        self._queue.append((script, script_args_builder))
        return len(self._queue) - 1  # Job id is the index in the queue

    def is_queue_full(self) -> bool:
        """
        Simple job systems never fill up as long as they have memory, queue is never full.
        :return:
        """
        return False

    def run_queued_jobs(self):
        """
        Actually run the jobs.
        Does so in a subprocess to maintain a clean state between tasks.
        :return: void
        """
        if len(self._queue) > 0:
            # work out if the current python is within a conda environment or a python env
            run_args = []
            if 'CONDA_DEFAULT_ENV' in os.environ:
                logging.getLogger(__name__).info("Using Anaconda environment {0}".format(
                    os.environ['CONDA_DEFAULT_ENV']))
                run_args = ['conda', 'run', '-n', os.environ['CONDA_DEFAULT_ENV']]
            # TODO: Handle virtualenv
            # elif 'VIRTUAL_ENV' in os.environ:
            #     run_args = []
            else:
                logging.getLogger(__name__).info("Using default python")

            for idx, (script_path, script_args_builder) in enumerate(self._queue):
                if is_id(script_path):
                    # This is just a task id, run the task directly
                    logging.getLogger(__name__).info("Running job {0} of {1} ...".format(idx, len(self._queue)))
                    arvet.batch_analysis.scripts.run_task.main(script_path, self._config_path)
                else:
                    logging.getLogger(__name__).info("Running job {0} of {1} in subprocess...".format(
                        idx, len(self._queue)))
                    script_args = script_args_builder(self._config_path)
                    subprocess.run(run_args + ['python', script_path] + script_args, cwd=os.getcwd())
            self._queue = []


def is_id(val: str) -> bool:
    """
    Is a given string a bson id. It should be a string of hex digits
    :param val:
    :return:
    """
    return re.fullmatch('[0-9a-f]+', val) is not None
