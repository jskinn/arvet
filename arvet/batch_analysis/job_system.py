# Copyright (c) 2017, John Skinner
import abc
import typing
import bson
import arvet.util.dict_utils as du
from arvet.batch_analysis.task import Task
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask
from arvet.batch_analysis.tasks.compare_trials_task import CompareTrialTask


class IJobSystem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run_task(self, task: Task) -> bool:
        """
        Queue a particular task to run.
        It doesn't necessarily have to start running immediately, but the job system must update the job state
        when it is started.
        Task start is deferred to allow the job system to make decisions about what tasks it is running based on
        the full set.
        Does not necessarily have to run all tasks passed to it.
        :param task: The task object to run
        :return: True if the job is expected to run, false if it will not (such as if the queue is full).
        """
        pass

    @abc.abstractmethod
    def run_script(self, script: str, script_args_builder: typing.Callable[..., typing.List[str]],
                   job_name: str = "", num_cpus: int = 1, num_gpus: int = 0,
                   memory_requirements: str = '3GB', expected_duration: str = '1:00:00') -> bool:
        """
        Run a python script that is not a task on this job system
        :param script: The path to the script to run
        :param script_args_builder: A function that returns a list of command line arguments, as strings
        :param job_name: A unique name for the job
        :param num_cpus: The number of CPUs required
        :param num_gpus: The number of GPUs required
        :param memory_requirements: The required amount of memory
        :param expected_duration: The duration given for the job to run
        :return: True if the
        """
        pass

    @abc.abstractmethod
    def is_queue_full(self) -> bool:
        """
        Some job systems may only allow a finite number of jobs at once.
        Once we have hit this limit, we expect run_script and run_task to no longer run the scripts/tasks given.
        In this case, we should stop providing them.
        :return:
        """
        pass


class JobSystem(metaclass=abc.ABCMeta):

    def __init__(self, config):
        self._node_id = config.get('node_id', type(self).__name__)

        if config is not None and 'task_config' in config:
            task_config = dict(config['task_config'])
        else:
            task_config = {}

        # Default configuration. Also serves as an exemplar configuration argument
        du.defaults(task_config, {
            'allow_import_dataset': True,
            'allow_run_system': True,
            'allow_measure': True,
            'allow_trial_comparison': True
        })
        self._allow_import_dataset = bool(task_config['allow_import_dataset'])
        self._allow_run_system = bool(task_config['allow_run_system'])
        self._allow_measure = bool(task_config['allow_measure'])
        self._allow_trial_comparison = bool(task_config['allow_trial_comparison'])

    @property
    def node_id(self) -> str:
        """
        All job systems should have a node id, controlled by the configuration.
        The idea is that different job systems on different computers have different
        node ids, so that we can track which system is supposed to be running which job id.
        :return:
        """
        return self._node_id

    @abc.abstractmethod
    def can_generate_dataset(self, simulator: bson.ObjectId, config: dict) -> bool:
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
    def is_job_running(self, job_id: int) -> bool:
        """
        Is the specified job id currently running through this job system.
        This is used by the task manager to work out which jobs have failed without notification, to reschedule them
        :param job_id: The integer job id to check
        :return: True if the job is currently running on this node
        """
        pass

    @abc.abstractmethod
    def run_task(self, task: Task) -> bool:
        """
        Queue a particular task to run.
        It doesn't necessarily have to start running immediately, but the job system must update the job state
        when it is started.
        Task start is deferred to allow the job system to make decisions about what tasks it is running based on
        the full set.
        Does not necessarily have to run all tasks passed to it.
        :param task: The task object to run
        :return: True if the job is expected to run, false if it will not (such as if the queue is full).
        """
        pass

    @abc.abstractmethod
    def run_script(self, script: str, script_args_builder: typing.Callable[..., typing.List[str]],
                   job_name: str = "", num_cpus: int = 1, num_gpus: int = 0,
                   memory_requirements: str = '3GB', expected_duration: str = '1:00:00') -> bool:
        """
        Run a python script that is not a task on this job system
        :param script: The path to the script to run
        :param script_args_builder: A function that returns a list of command line arguments, as strings
        :param job_name: A unique name for the job
        :param num_cpus: The number of CPUs required
        :param num_gpus: The number of GPUs required
        :param memory_requirements: The required amount of memory
        :param expected_duration: The duration given for the job to run
        :return: True if the
        """
        pass

    @abc.abstractmethod
    def is_queue_full(self) -> bool:
        """
        Some job systems may only allow a finite number of jobs at once.
        Once we have hit this limit, we expect run_script and run_task to no longer run the scripts/tasks given.
        In this case, we should stop providing them.
        :return:
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

    def can_run_task(self, task: Task) -> bool:
        """
        Check if the job system is allowed to run tasks of a partiular type.
        Task types are blacklisted here, unknown types will default to true.
        Override this for custom tasks
        :param task:
        :return:
        """
        if (
                (isinstance(task, ImportDatasetTask) and not self._allow_import_dataset) or
                (isinstance(task, RunSystemTask) and not self._allow_run_system) or
                (isinstance(task, MeasureTrialTask) and not self._allow_measure) or
                (isinstance(task, CompareTrialTask) and not self._allow_trial_comparison)
        ):
            return False
        return True
