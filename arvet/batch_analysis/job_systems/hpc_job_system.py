# Copyright (c) 2017, John Skinner
from os import getcwd
import logging
import typing
import subprocess
import re
import bson
from pathlib import Path
from operator import attrgetter
from functools import partial
import arvet.batch_analysis.job_system
from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
import arvet.batch_analysis.scripts.run_task


# Basic structure for job arguments
JOB_ARGS_TEMPLATE = """
#PBS -N {name}
#PBS -l walltime={time}
#PBS -l mem={mem}
#PBS -l ncpus={cpus}
#PBS -l cpuarch=avx2
"""


# Some additional arguments in the script for using GPUs
GPU_ARGS_TEMPLATE = """
#PBS -l ngpus={gpus}
#PBS -l gputype=M40
#PBS -l cputype=E5-2680v4
"""


SSH_TUNNEL_PREFIX = """
ssh -nN -i {ssh_key} -L {local_port}:localhost:27017 {username}@{hostname} &
echo $! > {job_folder}/{job_name}-{local_port}.ssh.pid
"""


SSH_TUNNEL_SUFFIX = """
cat {job_folder}/{job_name}-{local_port}.ssh.pid | xargs kill
rm {job_folder}/{job_name}-{local_port}.ssh.pid
"""


class HPCJobSystem(arvet.batch_analysis.job_system.JobSystem):
    """
    A job system using HPC to run tasks.

    """

    def __init__(self, config: dict, config_file: str):
        """
        Takes configuration parameters in a dict with the following format:
        {
            'node_id': 'name_of_job_system_node'
            'environment': 'path-to-activate-script'
            'job_location: 'folder-to-create-jobs'      # Default ~
            'job_name_prefix': 'prefix-to-job-names'    # Default ''
            'max_jobs': int                             # Default no limit
            'ssh_tunnel': {                             # No tunnel if omitted
                'hostname': The host to ssh to
                'username': The username to connect with
                'ssh_key':  Path to the SSH key to connect with
                'min_port': The minimum local port to use
                'max_port': The highest local port to use
            }
        }
        :param config: A dict of configuration parameters
        """
        super().__init__(config)
        self._config_path = Path(config_file).expanduser().resolve()

        # Work out what execution environment to use, virtualenv or conda
        self._environment = config.get('environment', None)
        if self._environment is not None:
            self._environment = Path(self._environment).expanduser().resolve()
        self._job_folder = config.get('job_location', '~')
        self._job_folder = Path(self._job_folder).expanduser().resolve()
        self._name_prefix = config.get('job_name_prefix', '')
        self._max_jobs = max(1, int(config['max_jobs'])) if 'max_jobs' in config else None

        # Configure the job to set up an ssh tunnel before running.
        ssh_tunnel_config = config.get('ssh_tunnel', {})
        self._ssh_host = ssh_tunnel_config.get('hostname', None)
        self._ssh_key = ssh_tunnel_config.get('ssh_key', None)
        self._ssh_key = Path(self._ssh_key).expanduser().resolve() if self._ssh_key is not None else None
        self._ssh_username = ssh_tunnel_config.get('username', None)
        self._ssh_min_port = int(ssh_tunnel_config.get('min_port', 5000))
        self._ssh_max_port = int(ssh_tunnel_config.get('max_port', 65535))
        self._use_ssh = bool(
            self._ssh_host is not None and
            self._ssh_key is not None and
            self._ssh_username is not None
        )

        if self._use_ssh and (self._max_jobs is None or (self._ssh_max_port - self._ssh_min_port + 1) < self._max_jobs):
            self._max_jobs = self._ssh_max_port - self._ssh_min_port + 1

        self._checked_running_jobs = False
        self._ssh_current_port = self._ssh_min_port
        self._tasks_to_run = []
        self._import_dataset_tasks = []
        self._scripts_to_run = []

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
        2315056.pbs       auto_task_1      user                     0 Q quick

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

    def run_task(self, task: Task):
        """
        Run a particular task
        :param task: The the task to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        if self.can_run_task(task):
            if isinstance(task, ImportDatasetTask):
                self._import_dataset_tasks.append(task)
            else:
                self._tasks_to_run.append(task)

    def run_script(
            self,
            script: typing.Union[str, Path],
            script_args_builder: typing.Callable[..., typing.List[str]],
            job_name: str = "",
            num_cpus: int = 1, num_gpus: int = 0,
            memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
    ):
        """
        Run a script that is not a task on this job system
        :param script: The path to the script to run
        :param script_args_builder: A lambda that returns list of command line arguments, as strings
        :param job_name: A unique name to use for the job.
        :param num_cpus: The number of CPUs required
        :param num_gpus: The number of GPUs required
        :param memory_requirements: The required amount of memory
        :param expected_duration: The duration given for the job to run
        :return: The job id if the job has been started correctly, None if failed.
        """
        self._scripts_to_run.append((
            script, script_args_builder, job_name, num_cpus, num_gpus, memory_requirements, expected_duration
        ))

    def run_queued_jobs(self):
        """
        Run queued jobs.
        We're doing something a little complex:
        HPC will run jobs in parallel. Most tasks are fine to run in parallel,
        however HDF5 does not support concurrent write, so ImportDatasetTasks must be run sequentially instead
        So we have three groups of jobs:
        - Scripts, which are run parallel
        - Tasks that are not ImportDatasetTasks, which are also run parallel
        - ImportDatasetTasks, which are run all together in a single job
        :return:
        """
        # Check if there are any import dataset tasks already running on this node
        existing_import_tasks_count = ImportDatasetTask.objects.raw({
            'node_id': self.node_id,
            'state': JobState.RUNNING.name
        }).count()
        if existing_import_tasks_count >= 1:
            # There are import dataset tasks running on this node, we can't do anything until they finish.
            return

        # Make a single job for all the import dataset tasks
        # We merge most of the requirements like num cpus as maximum across all the tasks
        # expected duration is the sum of all the durations.
        if len(self._import_dataset_tasks) > 0:
            logging.getLogger(__name__).info(
                "Submitting script for {0} import jobs".format(len(self._import_dataset_tasks)))
            memory_requirements = max(
                parse_memory_requirements(task.memory_requirements)
                for task in self._import_dataset_tasks
            )
            memory_requirements = max(1, memory_requirements // (1024 * 1024))
            job_id = self._create_and_run_script(
                scripts=[(
                    Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
                    partial(task_args_builder, task, True)
                ) for task in self._import_dataset_tasks],
                job_name="import_{0}_datasets".format(len(self._import_dataset_tasks)),
                num_cpus=max(task.num_cpus for task in self._import_dataset_tasks),
                num_gpus=max(task.num_gpus for task in self._import_dataset_tasks),
                memory_requirements="{0}GB".format(memory_requirements),
                expected_duration=merge_expected_durations(
                    task.expected_duration for task in self._import_dataset_tasks)
            )
            if job_id is not None:
                # The one job is running all the imports, mark that fact on all the tasks
                for task in self._import_dataset_tasks:
                    task.mark_job_started(self.node_id, job_id)
                    task.save()
            self._import_dataset_tasks = []
        else:
            # There are no import tasks, the HDF5 file is not changing
            # Run scripts in parallel
            logging.getLogger(__name__).info("Submitting {0} scripts to HPC".format(len(self._scripts_to_run)))
            for (
                    script,
                    script_args_builder,
                    job_name,
                    num_cpus,
                    num_gpus,
                    memory_requirements,
                    expected_duration
            ) in self._scripts_to_run:
                self._create_and_run_script(
                    scripts=[(script, script_args_builder)],
                    job_name=job_name,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    memory_requirements=memory_requirements,
                    expected_duration=expected_duration
                )
            self._scripts_to_run = []

            # Submit jobs for tasks that don't import, starting with those that have failed the least
            logging.getLogger(__name__).info("Submitting {0} tasks to HPC".format(len(self._tasks_to_run)))
            for task in sorted(self._tasks_to_run, key=attrgetter('failure_count')):
                job_id = self._create_and_run_script(
                    scripts=[(
                        Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
                        partial(task_args_builder, task, False)
                    )],
                    job_name=task.get_unique_name(),
                    num_cpus=task.num_cpus,
                    num_gpus=task.num_gpus,
                    memory_requirements=task.memory_requirements,
                    expected_duration=task.expected_duration
                )
                if job_id is not None:
                    task.mark_job_started(self.node_id, job_id)
                    task.save()
            self._tasks_to_run = []

    def _create_and_run_script(
            self,
            scripts: typing.Collection[typing.Tuple[typing.Union[str, Path], typing.Callable[..., typing.List[str]]]],
            job_name: str = "",
            num_cpus: int = 1, num_gpus: int = 0,
            memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
    ) -> typing.Union[int, None]:
        """
        Actually create and submit a job, which may run one or more actual scripts.
        Only ever counts as one toward max_jobs
        :param scripts: A
        :param job_name:
        :param num_cpus:
        :param num_gpus:
        :param memory_requirements:
        :param expected_duration:
        :return:
        """
        if len(scripts) <= 0:
            # No point in creating a job that doesn't run any scripts
            return None

        # Optionally limit the number of jobs
        if self._max_jobs is not None:
            # If we haven't yet, get the current number of running jobs,
            # so we don't double up by running this repeatedly
            if not self._checked_running_jobs:
                result = subprocess.run(['qjobs'], stdout=subprocess.PIPE, universal_newlines=True)
                re_match = re.search('(\\d+) running jobs found', result.stdout)
                if re_match is not None:
                    num_jobs = int(re_match.groups(default='0')[0])
                    self._max_jobs -= num_jobs
                    self._checked_running_jobs = True

            # Don't submit more than the max jobs, if a limit is set
            if self._max_jobs <= 0:
                # Cannot submit any more jobs
                logging.getLogger(__name__).info("Failed to submit job, job limit reached")
                return None
            self._max_jobs -= 1

        # Choose a job name and a unique file
        job_name = self._name_prefix + job_name
        job_file_path = self._job_folder / (job_name + '.sub')
        offset = 0
        while job_file_path.exists():
            offset += 1
            job_file_path = self._job_folder / (job_name + '_{0}.sub'.format(offset))

        lines = ['#!/bin/bash -l']
        # basic job meta-information
        if not isinstance(expected_duration, str) or not re.match('^[0-9]+:[0-9]{2}:[0-9]{2}$', expected_duration):
            expected_duration = '1:00:00'
        if not isinstance(memory_requirements, str) or not re.match('^[0-9]+[TGMK]B$', memory_requirements):
            memory_requirements = '3GB'
        lines.append(JOB_ARGS_TEMPLATE.format(
            name=job_name,
            time=expected_duration,
            mem=memory_requirements,
            cpus=num_cpus
        ).strip())

        # Additional args based on the number of GPUs
        if num_gpus > 0:
            lines.append(GPU_ARGS_TEMPLATE.format(gpus=num_gpus).strip())
        elif int(memory_requirements.rstrip('MGB')) > 125:
            lines.append('#PBS -l cputype=E5-2680v3')

        # Actually assemble the script commands, line group by line group
        if self._environment is not None:
            lines.append('source ' + quote(str(self._environment)))

        # change to the current working dir
        lines.append('cd {0}'.format(quote(getcwd())))

        # Activate an SSH tunnel, if required
        port = None
        if self._use_ssh:
            # Find a port *we* are not using
            port = self._ssh_current_port
            while port <= self._ssh_max_port and any(True for _ in self._job_folder.glob('*-{0}.ssh.pid'.format(port))):
                port += 1
            self._ssh_current_port = port + 1

            lines.append(SSH_TUNNEL_PREFIX.format(
                job_name=job_name,
                username=self._ssh_username,
                ssh_key=self._ssh_key,
                local_port=port,
                hostname=self._ssh_host,
                job_folder=self._job_folder
            ).strip())

        # Add the actual script commands
        for script, script_args_builder in scripts:
            script_args = script_args_builder(self._config_path, port)
            lines.append('python {script} {args}'.format(
                script=quote(str(script)),
                args=' '.join([quote(arg) for arg in script_args])
            ))

        # Add commands for closing the SSH tunnel, if we have one
        if self._use_ssh:
            lines.append(SSH_TUNNEL_SUFFIX.format(
                job_name=job_name, local_port=port, job_folder=self._job_folder).strip())

        # Clean up the job file when we're done
        lines.append("rm {0}".format(job_file_path))

        # Write the job file
        with open(job_file_path, 'w+') as job_file:
            job_file.write('\n'.join(lines))

        logging.getLogger(__name__).info("Submitting job file {0}".format(job_file_path))
        result = subprocess.run(['qsub', job_file_path], stdout=subprocess.PIPE, universal_newlines=True)
        job_id = re.search('(\\d+)', result.stdout).group()
        return int(job_id)


def task_args_builder(task: Task, allow_write: bool, config, port: int = None):
    """
    Make the command line arguments for running a task
    Will use partials to bind the values of task and allow_write
    :param task:
    :param allow_write:
    :param config:
    :param port:
    :return:
    """
    args = ['--config', str(config)]
    if port is not None:
        args += ['--mongodb_port', str(port)]
    if allow_write:
        args.append('--allow_write')
    args.append(str(task.pk))
    return args


def parse_memory_requirements(memory: str):
    """
    Turn a memory requirements string to a size in KB
    :param memory:
    :return:
    """
    memory = memory.upper()
    if memory.endswith('GB'):
        return 1024 * 1024 * int(memory.rstrip('GMKB'))
    elif memory.endswith('MB'):
        return 1024 * int(memory.rstrip('GMKB'))
    return int(memory.rstrip('GMKB'))


def merge_expected_durations(durations: typing.Iterable[str]) -> str:
    """
    Join together the estimated times of multiple tasks into a combined time
    :param durations:
    :return:
    """
    hours = 0
    minutes = 0
    seconds = 0
    for duration in durations:
        parts = duration.split(':')
        if len(parts) >= 3:
            hours += int(parts[0])
            minutes += int(parts[1])
            seconds += int(parts[2])
    minutes += seconds // 60
    seconds = seconds % 60
    hours += minutes // 60
    minutes = minutes % 60
    return "{0:02}:{1:02}:{2:02}".format(hours, minutes, seconds)


def quote(string: str) -> str:
    """
    Wrap a string with quotes iff it contains a space. Used for interacting with command line scripts.
    :param string:
    :return:
    """
    if ' ' in string:
        return '"' + string + '"'
    return string
