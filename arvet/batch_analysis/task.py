# Copyright (c) 2017, John Skinner
import abc
import enum
import bson
import pymodm
import pymodm.fields as fields
from arvet.database.pymodm_abc import ABCModelMeta
from arvet.database.enum_field import EnumField
from arvet.config.path_manager import PathManager


class JobState(enum.Enum):
    UNSTARTED = 0
    RUNNING = 1
    DONE = 2


class TaskType(enum.Enum):
    """
    These are the 8 kinds of tasks in this system.
    They are things that are done asynchronously, and take significant time.
    """
    GENERATE_DATASET = 0
    IMPORT_DATASET = 1
    TRAIN_SYSTEM = 2
    IMPORT_SYSTEM = 3
    TEST_SYSTEM = 4
    BENCHMARK_RESULT = 5
    COMPARE_TRIALS = 6
    COMPARE_BENCHMARKS = 7


class Task(pymodm.MongoModel, metaclass=ABCModelMeta):
    """
    A Task entity tracks performing a specific task.
    The only two properties you should use here are 'is_finished' and 'result',
    to check if your tasks are done and get their output.

    NEVER EVER CREATE THESE OUTSIDE TASK MANAGER.
    Instead, call the appropriate get method on task manager to get a new task instance.

    NEVER EVER CHANGE STATE MANUALLY
    That is what mark_job_complete is for
    """
    state = EnumField(JobState, required=True)
    node_id = fields.CharField(blank=True)
    job_id = fields.IntegerField(blank=True)
    num_cpus = fields.IntegerField(default=1, min_value=1)
    num_gpus = fields.IntegerField(default=0, min_value=0)
    memory_requirements = fields.CharField(default='3GB')
    expected_duration = fields.CharField(default='1:00:00')

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the identifier for this task
        :return: The object id for this task, use for querying
        """
        return self._id

    @property
    def is_unstarted(self) -> bool:
        """
        Is the job unstarted. This is used internally by TaskManager to choose which new tasks to queue.
        You shouldn't need to use it.
        :return:
        """
        return JobState.UNSTARTED == self.state

    @property
    def is_running(self) -> bool:
        """
        Is the job currently running?
        Used to avoid scheduling jobs more than once
        :return: True iff the task is currently recorded as running
        """
        return JobState.RUNNING == self.state

    @property
    def is_finished(self) -> bool:
        """
        Is the job already done?
        Experiments should use this to check if result will be set.
        :return: True iff the task has been completed
        """
        return JobState.DONE == self.state

    @abc.abstractmethod
    def run_task(self, path_manager: PathManager) -> None:
        """
        Actually perform the task.
        Different subtypes do different things.
        :param path_manager: A path manager, to resolve file system paths
        :return:
        """
        pass

    @abc.abstractmethod
    def get_unique_name(self) -> str:
        """
        Get a pretty name for this task
        :return:
        """
        pass

    def mark_job_started(self, node_id: str, job_id: int) -> None:
        """
        Mark the job as having been started on a particular node, with a particular job id.
        Only works if the task is unstarted, can't start a complete job
        :param node_id: The id of the node running the job
        :param job_id: The id of the job on the node, so we can check it is still running
        :return: void
        """
        if self.is_unstarted:
            self.state = JobState.RUNNING
            self.node_id = node_id
            self.job_id = job_id

    def change_job_id(self, node_id: str, job_id: int) -> None:
        """
        Change the node_id and job_id of this task. This is used if the task takes several jobs to complete.
        Only works while the job is running.
        :param node_id: The new node id running the job
        :param job_id: The new job id on the node
        :return: void
        """
        if self.is_running:
            self.node_id = node_id
            self.job_id = job_id

    def mark_job_complete(self) -> None:
        """
        Mark the task complete, with a particular result.
        :return:
        """
        if self.is_running:
            self.state = JobState.DONE
            self.node_id = None
            self.job_id = None

    def mark_job_failed(self) -> None:
        """
        A running task has failed, return it to unstarted state so we can try again.
        :return:
        """
        if self.is_running:
            self.state = JobState.UNSTARTED
            self.node_id = None
            self.job_id = None
