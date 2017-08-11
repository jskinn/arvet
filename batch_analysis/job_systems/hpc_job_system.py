import os
import logging
import subprocess
import re
import time
import batch_analysis.job_system
import task_import_dataset
import task_train_system
import task_run_system
import task_benchmark_result


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
            # Optional, will look for env used by current process if omitted
            'environment': 'path-to-virtualenv-activate'
            'job_location: 'folder-to-create-jobs'      # Default ~
            'job_name_prefix': 'prefix-to-job-names'    # Default ''
        }
        :param config: A dict of configuration parameters
        """
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
        self._queued_jobs = []

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

    def queue_generate_dataset(self, simulator_id, config, experiment=None, num_cpus=1, num_gpus=0,
                               memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Queue generating a synthetic dataset using a particular simulator
        and a particular configuration
        :param simulator_id: The id of the simulator to use to generate the dataset
        :param config: Configuration passed to the simulator to control the dataset generation
        :param experiment: The experiment this generated dataset is associated with, if any
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: void
        """
        return False

    def queue_import_dataset(self, module_name, path, experiment=None, num_cpus=1, num_gpus=0,
                             memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Create a HPC job to import an image dataset into the image dataset
        :param module_name: The python module to use to do the import
        :param path: The directory to import the dataset from
        :param experiment: The experiment associated with this import, if any
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: True iff the job was queued
        """
        return self.create_job('import', task_import_dataset.__file__, str(module_name), str(path), experiment,
                               num_cpus, num_gpus, memory_requirements, expected_duration)

    def queue_train_system(self, trainer_id, trainee_id, experiment=None, num_cpus=1, num_gpus=0,
                           memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Use the job system to train a system with a particular image source.
        Internally calls the 'run_script' function, above, with the "task_run_system" in the root of this project
        TODO: find a better way to get the path of the script
        :param trainer_id: The id of the trainer doing the training
        :param trainee_id: The id of the trainee being trained
        :param experiment: The experiment associated with this run, if any
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: True iff the job was queued
        """
        return self.create_job('train', task_train_system.__file__, str(trainer_id), str(trainee_id), experiment,
                               num_cpus, num_gpus, memory_requirements, expected_duration)

    def queue_run_system(self, system_id, image_source_id, experiment=None, num_cpus=1, num_gpus=0,
                         memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Use the job system to run a system with a particular image source.
        Internally calls the 'run_script' function, above, with the "task_run_system" in the root of this project
        TODO: find a better way to get the path of the script
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param experiment: The experiment associated with this run, if any
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: True iff the job was queued
        """
        return self.create_job('run', task_run_system.__file__, str(system_id), str(image_source_id), experiment,
                               num_cpus, num_gpus, memory_requirements, expected_duration)

    def queue_benchmark_result(self, trial_id, benchmark_id, experiment=None, num_cpus=1, num_gpus=0,
                               memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Use the job system to benchmark a particular trial result.
        Uses the 'run_script' function, above
        :param trial_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param experiment: The experiment this is associated with, if any
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: True iff the job was queued
        """
        return self.create_job('benchmark', task_benchmark_result.__file__, str(trial_id), str(benchmark_id),
                               experiment, num_cpus, num_gpus, memory_requirements, expected_duration)

    def push_queued_jobs(self):
        """
        Actually add the queued jobs to the HPC job queue
        :return:
        """
        for job_file in self._queued_jobs:
            logging.getLogger(__name__).info("Submitting job file {0}".format(job_file))
            subprocess.call(['qsub', job_file])

    def create_job(self, type_, script_path, arg1, arg2, experiment=None, num_cpus=1, num_gpus=0,
                   memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Create a new HPC job file, ready to be submitted to the queue.
        It's basically the same for all types of job, so we have this helper function to do all the work.

        :param type_: The type of job, used in the job name
        :param script_path: The path of the python file to run
        :param arg1: The first argument to the script, as a string
        :param arg2: The second argument to the script
        :param experiment: The experiment id to use, or None if no experiment
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: void
        """
        # Job meta-information
        name = "{0}-{1}".format(type_, time.time())
        name = self._name_prefix + name.replace(' ', '-').replace('/', '-').replace('.', '-')
        if not isinstance(expected_duration, str) or not re.match('^[0-9]+:[0-9]{2}:[0-9]{2}$', expected_duration):
            expected_duration = '1:00:00'
        if not isinstance(memory_requirements, str) or not re.match('^[0-9]+[TGMK]B$', memory_requirements):
            memory_requirements = '3GB'
        job_params = ""
        if num_gpus > 0:
            job_params += GPU_ARGS_TEMPLATE.format(gpus=num_gpus)
        env = ('source ' + quote(self._virtual_env)) if self._virtual_env is not None else ''

        # Parameter args
        args = quote(arg1) + ' ' + quote(arg2)  # Quotes around the args to handle spaces
        if experiment is not None:
            args += ' ' + str(experiment)
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
                args=args
            ))
        logging.getLogger(__name__).info("Queueing {0} job in file {1}".format(type_, job_file_path))
        self._queued_jobs.append(job_file_path)
        return True


def quote(string):
    if ' ' in string:
        return '"' + string + '"'
    return string
