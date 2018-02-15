# Copyright (c) 2017, John Skinner
import os
import logging
import typing
import subprocess
import re
import time
import bson
import arvet.batch_analysis.job_system
import arvet.batch_analysis.scripts.run_task


# This is the template for python scripts run by the hpc
JOB_TEMPLATE = """#!/bin/bash -l
#PBS -N {name}
#PBS -l walltime={time}
#PBS -l mem={mem}
#PBS -l ncpus={cpus}
#PBS -l cpuarch=avx
{job_params}
{env}
cd {working_directory}
python {script} {args}
"""


# Some additional arguments in the script for using GPUs
GPU_ARGS_TEMPLATE = """
#PBS -l ngpus={gpus}
#PBS -l gputype=M40
#PBS -l cputype=E5-2680v4
"""


class HPCJobSystem(arvet.batch_analysis.job_system.JobSystem):
    """
    A job system using HPC to run tasks.

    """

    def __init__(self, config: dict):
        """
        Takes configuration parameters in a dict with the following format:
        {
            'node_id': 'name_of_job_system_node'
            # Optional, will look for env used by current process if omitted
            'environment': 'path-to-virtualenv-activate'
            'job_location: 'folder-to-create-jobs'      # Default ~
            'job_name_prefix': 'prefix-to-job-names'    # Default ''
        }
        :param config: A dict of configuration parameters
        """
        self._node_id = config['node_id'] if 'node_id' in config else 'hpc-job-system'
        self._virtual_env = None
        if 'environment' in config:
            self._virtual_env = config['environment']
        elif 'VIRTUAL_ENV' in os.environ:
            # No configured virtual environment, but this process has one, use it
            self._virtual_env = os.path.join(os.environ['VIRTUAL_ENV'], 'bin/activate')
        self._virtual_env = os.path.expanduser(self._virtual_env)
        self._job_folder = config['job_location'] if 'job_location' in config else '~'
        self._job_folder = os.path.expanduser(self._job_folder)
        self._name_prefix = config['job_name_prefix'] if 'job_name_prefix' in config else ''

    @property
    def node_id(self) -> str:
        """
        All job systems should have a node id, controlled by the configuration.
        The idea is that different job systems on different computers have different
        node ids, so that we can track which system is supposed to be running which job id.
        :return:
        """
        return self._node_id

    def can_generate_dataset(self, simulator: bson.ObjectId, config: dict) -> bool:
        """
        Can this job system generate synthetic datasets.
        HPC cannot generate datasets, because it is a server with
        no X session
        :param simulator: The simulator id that will be doing the generation
        :param config: Configuration passed to the simulator at run time
        :return: True iff the job system can generate datasets. HPC cannot.
        """
        return False

    def is_job_running(self, job_id: int) -> bool:
        """
        Is the specified job id currently running through this job system.
        This is used by the task manager to work out which jobs have failed without notification, to reschedule them.
        For the HPC, a job is valid based on the output of the command 'qstat'
        A running job id produced output like:
        Job id            Name             User              Time Use S Queue
        ----------------  ---------------- ----------------  -------- - -----
        2315056.pbs       jrs_auto_task_1  n9520864                 0 Q quick

        whereas a non-running job produces:
        qstat: Unknown Job Id 2315.pbs
        an invalid job produces:
        qstat: illegally formed job identifier: 231512525
        A finished job produces:
        qstat: 2338916.pbs Job has finished, use -x or -H to obtain historical job information

        :param job_id: The integer job id to check
        :return: True if the job is currently running on this node
        """
        result = subprocess.run(['qstat', str(int(job_id))], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        output = result.stdout.lower()  # Case insensitive
        return 'unknown job id' not in output and 'job has finished' not in output

    def run_task(self, task_id: bson.ObjectId, num_cpus: int = 1, num_gpus: int = 0, memory_requirements: str = '3GB',
                 expected_duration: str = '1:00:00') -> int:
        """
        Run a particular task
        :param task_id: The id of the task to run
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: The job id if the job has been started correctly, None if failed.
        """
        return self.run_script(
            script=arvet.batch_analysis.scripts.run_task.__file__,
            script_args=[str(task_id)],
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_requirements=memory_requirements,
            expected_duration=expected_duration
        )

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
        # Job meta-information
        # TODO: We need better job names
        name = self._name_prefix + "auto_task_{0}".format(time.time()).replace('.', '-')
        if not isinstance(expected_duration, str) or not re.match('^[0-9]+:[0-9]{2}:[0-9]{2}$', expected_duration):
            expected_duration = '1:00:00'
        if not isinstance(memory_requirements, str) or not re.match('^[0-9]+[TGMK]B$', memory_requirements):
            memory_requirements = '3GB'
        job_params = ""
        if num_gpus > 0:
            job_params = GPU_ARGS_TEMPLATE.format(gpus=num_gpus)
        elif int(memory_requirements.rstrip('MGB')) > 125:
            job_params = '#PBS -l cputype=E5-2680v3'
        env = ('source ' + quote(self._virtual_env)) if self._virtual_env is not None else ''

        # Parameter args
        job_file_path = os.path.join(self._job_folder, name + '.sub')
        with open(job_file_path, 'w+') as job_file:
            job_file.write(JOB_TEMPLATE.format(
                name=name,
                time=expected_duration,
                mem=memory_requirements,
                cpus=int(num_cpus),
                job_params=job_params,
                env=env,
                working_directory=quote(os.getcwd()),
                script=quote(script),
                args=' '.join([quote(arg) for arg in script_args])
            ))

        logging.getLogger(__name__).info("Submitting job file {0}".format(job_file_path))
        result = subprocess.run(['qsub', job_file_path], stdout=subprocess.PIPE, universal_newlines=True)
        job_id = re.search('(\d+)', result.stdout).group()
        return int(job_id)

    def run_queued_jobs(self):
        """
        Run queued jobs.
        Since we've already sent the jobs to the PBS job system, don't do anything.
        :return:
        """
        pass


def quote(string: str) -> str:
    """
    Wrap a string with quotes iff it contains a space. Used for interacting with command line scripts.
    :param string:
    :return:
    """
    if ' ' in string:
        return '"' + string + '"'
    return string
