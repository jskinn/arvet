# Copyright (c) 2017, John Skinner
import arvet.database.entity
import enum


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


class Task(arvet.database.entity.Entity, metaclass=arvet.database.entity.AbstractEntityMetaclass):
    """
    A Task entity tracks performing a specific task.
    The only two properties you should use here are 'is_finished' and 'result',
    to check if your tasks are done and get their output.

    NEVER EVER CREATE THESE OUTSIDE TASK MANAGER.
    Instead, call the appropriate get method on task manager to get a new task instance.
    """

    def __init__(self, state=JobState.UNSTARTED, node_id=None, job_id=None, result=None, num_cpus=1, num_gpus=0,
                 memory_requirements='3GB', expected_duration='1:00:00', id_=None):
        super().__init__(id_=id_)
        self._state = JobState(state)
        self._node_id = node_id
        self._job_id = int(job_id) if job_id is not None else None
        self._result = result
        self._num_cpus = int(num_cpus)
        self._num_gpus = int(num_gpus)
        self._memory_requirements = memory_requirements
        self._expected_duration = expected_duration
        self._updates = {}

    @property
    def is_finished(self):
        """
        Is the job already done?
        Experiments should use this to check if result will be set.
        :return: True iff the task has been completed
        """
        return JobState.DONE == self._state

    @property
    def result(self):
        """
        Get the result from running this task, usually a database id.
        :return:
        """
        return self._result

    @property
    def node_id(self):
        """
        Get the id of the job system node running this task if it is running.
        You should not need this property, TaskManager is handling it.
        :return: String node id from the job system configuration.
        """
        return self._node_id

    @property
    def job_id(self):
        """
        Get the id of the job with the job system.
        You should not need this property, TaskManager is handling it
        :return: Integer
        """
        return self._job_id

    @property
    def is_unstarted(self):
        """
        Is the job unstarted. This is used internally by TaskManager to choose which new tasks to queue.
        You shouldn't need to use it.
        :return:
        """
        return JobState.UNSTARTED == self._state

    @property
    def num_cpus(self):
        return self._num_cpus

    @property
    def num_gpus(self):
        return self._num_gpus

    @property
    def memory_requirements(self):
        return self._memory_requirements

    @property
    def expected_duration(self):
        return self._expected_duration

    def run_task(self, db_client):
        """
        Actually perform the task.
        Different subtypes do different things.
        :return:
        """
        pass

    def mark_job_started(self, node_id, job_id):
        if JobState.UNSTARTED == self._state:
            self._state = JobState.RUNNING
            self._node_id = node_id
            self._job_id = job_id
            if '$set' not in self._updates:
                self._updates['$set'] = {}
            self._updates['$set']['state'] = JobState.RUNNING.value
            self._updates['$set']['node_id'] = node_id
            self._updates['$set']['job_id'] = job_id
            # Don't unset the job id anymore, we're setting it to something else insted
            if '$unset' in self._updates:
                if 'node_id' in self._updates['$unset']:
                    del self._updates['$unset']['node_id']
                if 'job_id' in self._updates['$unset']:
                    del self._updates['$unset']['job_id']
                if self._updates['$unset'] == {}:
                    del self._updates['$unset']

    def mark_job_complete(self, result):
        if JobState.RUNNING == self._state:
            self._state = JobState.DONE
            self._result = result
            self._node_id = None
            self._job_id = None
            if '$set' not in self._updates:
                self._updates['$set'] = {}
            self._updates['$set']['state'] = JobState.DONE.value
            self._updates['$set']['result'] = result
            if '$unset' not in self._updates:
                self._updates['$unset'] = {}
            self._updates['$unset']['node_id'] = True
            self._updates['$unset']['job_id'] = True

            # Don't set the job id anymore, it's getting unset
            if 'node_id' in self._updates['$set']:
                del self._updates['$set']['node_id']
            if 'job_id' in self._updates['$set']:
                del self._updates['$set']['job_id']

    def mark_job_failed(self):
        if JobState.RUNNING == self._state:
            self._state = JobState.UNSTARTED
            self._node_id = None
            self._job_id = None
            if '$set' not in self._updates:
                self._updates['$set'] = {}
            self._updates['$set']['state'] = JobState.UNSTARTED.value
            if '$unset' not in self._updates:
                self._updates['$unset'] = {}
            self._updates['$unset']['node_id'] = True
            self._updates['$unset']['job_id'] = True

            # Don't set the job id anymore, it's getting unset
            if 'node_id' in self._updates['$set']:
                del self._updates['$set']['node_id']
            if 'job_id' in self._updates['$set']:
                del self._updates['$set']['job_id']

    def save_updates(self, collection):
        if self.identifier is None:
            s_task = self.serialize()
            id_ = collection.insert(s_task)
            self.refresh_id(id_)
        elif len(self._updates) > 0:
            collection.update({'_id': self.identifier}, self._updates)
        self._updates = {}

    def serialize(self):
        serialized = super().serialize()
        serialized['state'] = self._state.value
        serialized['num_cpus'] = self._num_cpus
        serialized['num_gpus'] = self._num_gpus
        serialized['memory_requirements'] = self._memory_requirements
        serialized['expected_duration'] = self._expected_duration
        if self._state:
            serialized['node_id'] = self.node_id
            serialized['job_id'] = self.job_id
        if self.is_finished:
            serialized['result'] = self.result
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'state' in serialized_representation:
            kwargs['state'] = serialized_representation['state']
        if 'num_cpus' in serialized_representation:
            kwargs['num_cpus'] = serialized_representation['num_cpus']
        if 'num_gpus' in serialized_representation:
            kwargs['num_gpus'] = serialized_representation['num_gpus']
        if 'memory_requirements' in serialized_representation:
            kwargs['memory_requirements'] = serialized_representation['memory_requirements']
        if 'expected_duration' in serialized_representation:
            kwargs['expected_duration'] = serialized_representation['expected_duration']
        if 'node_id' in serialized_representation:
            kwargs['node_id'] = serialized_representation['node_id']
        if 'job_id' in serialized_representation:
            kwargs['job_id'] = serialized_representation['job_id']
        if 'result' in serialized_representation:
            kwargs['result'] = serialized_representation['result']
        return super().deserialize(serialized_representation, db_client, **kwargs)
