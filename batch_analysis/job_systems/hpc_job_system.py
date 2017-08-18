import os
import logging
import subprocess
import re
import time
import batch_analysis.job_system
import run_task


# This is the template for python scripts run by the hpc
JOB_TEMPLATE = """#!/bin/bash -l
#PBS -N {name}
#PBS -l walltime={time}
#PBS -l mem={mem}
#PBS -l ncpus={cpus}
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


class HPCJobSystem(batch_analysis.job_system.JobSystem):
    """
    A job system using HPC to run tasks.

    """

    def __init__(self, config):
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
        Can this job system generate synthetic datasets.
        HPC cannot generate datasets, because it is a server with
        no X session
        :param simulator: The simulator id that will be doing the generation
        :param config: Configuration passed to the simulator at run time
        :return: True iff the job system can generate datasets. HPC cannot.
        """
        return False

    def is_job_running(self, job_id):
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
        and an invalid job:
        qstat: illegally formed job identifier: 231512525

        :param job_id: The integer job id to check
        :return: True if the job is currently running on this node
        """
        result = subprocess.run(['qstat', int(job_id)], stdout=subprocess.PIPE, universal_newlines=True)
        return 'Unknown Job Id' not in result.stdout    # TODO: Better distinguish here once we have example output

    def run_task(self, task_id, num_cpus=1, num_gpus=0, memory_requirements='3GB',
                 expected_duration='1:00:00'):
        """
        Run a particular task
        :param task_id: The id of the task to run
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
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
        env = ('source ' + quote(self._virtual_env)) if self._virtual_env is not None else ''

        # Parameter args
        script_path = run_task.__file__
        job_file_path = os.path.join(self._job_folder, name + '.sub')
        with open(job_file_path, 'w+') as job_file:
            job_file.write(JOB_TEMPLATE.format(
                name=name,
                time=expected_duration,
                mem=memory_requirements,
                cpus=int(num_cpus),
                job_params=job_params,
                env=env,
                working_directory=quote(os.path.dirname(script_path)),
                script=quote(script_path),
                args=str(task_id)
            ))

        logging.getLogger(__name__).info("Submitting job file {0}".format(job_file_path))
        result = subprocess.run(['qsub', job_file_path], stdout=subprocess.PIPE, universal_newlines=True)
        # TODO: Get some example output, I'm parsing on guesswork here
        job_id = re.search('(\d+)', result.stdout).group()
        return int(job_id)

    def run_queued_jobs(self):
        """
        Run queued jobs.
        Since we've already sent the jobs to the PBS job system, don't do anything.
        :return:
        """
        pass


def quote(string):
    if ' ' in string:
        return '"' + string + '"'
    return string
