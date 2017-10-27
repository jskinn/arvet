# Copyright (c) 2017, John Skinner
import abc
import collections
import typing
import bson
import batch_analysis.task_manager
import database.client
import database.entity
import util.database_helpers as dh


class Experiment(database.entity.Entity, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A model for an experiment. The role of the experiment is to decide which systems should be run with
    which datasets, and which benchmarks should be used to measure them.
    They form a collection of results for us to analyse and write papers from

    Fundamentally, an experiment is a bunch of groups of ids for core object types,
    which will be mixed and matched in trials and benchmarks to produce our desired results.
    The groupings of the ids are meaningful for the particular experiment, to convey some level
    of association or meta-data that we can't articulate or encapsulate in the image_metadata.
    """

    def __init__(self, trial_map: dict = None, result_map: dict = None, id_: typing.Union[bson.ObjectId, None] = None):
        super().__init__(id_=id_)
        self._trial_map = trial_map if trial_map is not None else {}
        self._result_map = result_map if result_map is not None else {}
        self._updates = {}

    @abc.abstractmethod
    def do_imports(self, task_manager: batch_analysis.task_manager.TaskManager,
                   db_client: database.client.DatabaseClient):
        """
        Perform imports for this experiment,
        This is where object creation should be performed (like creating benchmarks or untrained systems)
        We should also use the task manager to create import tasks, but not any other kinds of tasks.

        :param task_manager: The task manager, for creating import tasks (import datasets, generate datasets)
        :param db_client: The database client, to save declared objects that are too small for an import
        :return: void
        """
        pass

    @abc.abstractmethod
    def schedule_tasks(self, task_manager: batch_analysis.task_manager.TaskManager,
                       db_client: database.client.DatabaseClient):
        """
        This is where you schedule the core tasks to train a system, run a system, or perform a benchmark,
        :param task_manager: The the task manager used to create tasks
        :param db_client: The database client, so that we can deserialize entities and check compatibility
        :return: void
        """
        pass

    def save_updates(self, db_client: database.client.DatabaseClient):
        """
        Save accumulated changes to the database.
        If the object is not already in the database (the identifier is None),
        instead saves the whole object.
        :param db_client: The database client, allowing us to save.
        :return:
        """
        if self.identifier is None:
            # We're not in the database yet, serialize and save.
            s_experiment = self.serialize()
            id_ = db_client.experiments_collection.insert(s_experiment)
            self.refresh_id(id_)
        elif len(self._updates) > 0:
            # We have some updates stored, push them and clear the stored values.
            db_client.experiments_collection.update({'_id': self.identifier}, self._updates)
            self._updates = {}

    def plot_results(self, db_client: database.client.DatabaseClient):
        """
        Visualise the results from this experiment.
        Non-compulsory, but will be called from plot_results.py
        :param db_client:
        :return:
        """
        pass

    def schedule_all(self, task_manager: batch_analysis.task_manager.TaskManager,
                     db_client: database.client.DatabaseClient,
                     systems: typing.List[bson.ObjectId],
                     image_sources: typing.List[bson.ObjectId],
                     benchmarks: typing.List[bson.ObjectId]):
        """
        Schedule all combinations of running some list of systems with some list of image sources,
        and then benchmarking the results with some list of benchmarks.
        Uses is_image_source_appropriate and is_benchmark_appropriate to filter.
        Created results can be retrieved with get_trial_result and get_benchmark_result.

        :param task_manager: The task manager to perform scheduling
        :param db_client: The database client, to load the systems, image sources, etc..
        :param systems: The list of system ids to test
        :param image_sources: The list of image source ids to use
        :param benchmarks: The list of benchmark ids to measure the results
        :return: void
        """
        # Trial results will be collected as we go
        trial_results = set()

        # For each image dataset, run libviso with that dataset, and store the result in the trial map
        for image_source_id in image_sources:
            image_source = dh.load_object(db_client, db_client.image_source_collection, image_source_id)
            if image_source is None:
                continue
            for system_id in systems:
                system = dh.load_object(db_client, db_client.system_collection, system_id)
                if system is not None and system.is_image_source_appropriate(image_source):
                    task = task_manager.get_run_system_task(
                        system_id=system.identifier,
                        image_source_id=image_source.identifier,
                        expected_duration='8:00:00',
                        memory_requirements='12GB'
                    )
                    if not task.is_finished:
                        task_manager.do_task(task)
                    else:
                        trial_results.add(task.result)
                        self.store_trial_result(system_id, image_source_id, task.result)

        # Benchmark trial results
        for trial_result_id in trial_results:
            trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
            if trial_result is None:
                continue
            for benchmark_id in benchmarks:
                benchmark = dh.load_object(db_client, db_client.benchmarks_collection, benchmark_id)
                if benchmark is not None and benchmark.is_trial_appropriate(trial_result):
                    task = task_manager.get_benchmark_task(
                        trial_result_id=trial_result.identifier,
                        benchmark_id=benchmark.identifier,
                        expected_duration='6:00:00',
                        memory_requirements='6GB'
                    )
                    if not task.is_finished:
                        task_manager.do_task(task)
                    else:
                        self.store_benchmark_result(trial_result_id, benchmark_id, task.result)

    def store_trial_result(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId,
                           trial_result_id: bson.ObjectId):
        """
        Store the result of running a particular system with a particular image source,
        that is, a trial result against the system and image source that produced it.
        Call this if you schedule trials yourself, it will be called automatically as part of schedule_all
        :param system_id:
        :param image_source_id:
        :param trial_result_id:
        :return: void
        """
        if system_id not in self._trial_map:
            self._trial_map[system_id] = {}
        self._trial_map[system_id][image_source_id] = trial_result_id
        self._set_property('trial_map.{0}.{1}'.format(system_id, image_source_id), trial_result_id)

    def get_trial_result(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId)\
            -> typing.Union[bson.ObjectId, None]:
        """
        Get the trial result produced by running a given system with a given image source.
        Return None if the system has not been run with that image source.
        :param system_id: The id of the system
        :param image_source_id: The id of the image source
        :return: The id of the trial result, or None if the trial has not been performed
        """
        if system_id in self._trial_map and image_source_id in self._trial_map[system_id]:
            return self._trial_map[system_id][image_source_id]
        return None

    def store_benchmark_result(self, trial_result_id: bson.ObjectId, benchmark_id: bson.ObjectId,
                               benchmark_result_id: bson.ObjectId):
        """
        Store the result of measuring a particular trial with a particular benchmark.
        :param trial_result_id: The id of the trial result
        :param benchmark_id: The id of the benchmark used
        :param benchmark_result_id: The id of the benchmark result
        :return: void
        """
        if trial_result_id not in self._result_map:
            self._result_map[trial_result_id] = {}
        self._result_map[trial_result_id][benchmark_id] = benchmark_result_id
        self._set_property('result_map.{0}.{1}'.format(trial_result_id, benchmark_id), benchmark_result_id)

    def get_benchmark_result(self, trial_result_id: bson.ObjectId, benchmark_id: bson.ObjectId) \
            -> typing.Union[bson.ObjectId, None]:
        """
        Get the results for a particular trial, from a particular benchmark.
        :param trial_result_id: The id of the trial to get
        :param benchmark_id: The id of the benchmark used
        :return: The id of the result object, or None if the trial has not been measured.
        """
        if trial_result_id in self._result_map and benchmark_id in self._result_map[trial_result_id]:
            return self._result_map[trial_result_id][benchmark_id]
        return None

    def serialize(self):
        """
        Serialize the experiment for storage in the database
        :return:
        """
        serialized = super().serialize()
        serialized['trial_map'] = {str(sys_id): {str(source_id): trial_id
                                                 for source_id, trial_id in inner_map.items()}
                                   for sys_id, inner_map in self._trial_map.items()}
        serialized['result_map'] = {str(trial_id): {str(bench_id): res_id
                                                    for bench_id, res_id in inner_map.items()}
                                    for trial_id, inner_map in self._result_map.items()}
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trial_map' in serialized_representation:
            kwargs['trial_map'] = {bson.ObjectId(sys_id): {bson.ObjectId(source_id): trial_id
                                                           for source_id, trial_id in inner_map.items()}
                                   for sys_id, inner_map in serialized_representation['trial_map'].items()}
        if 'result_map' in serialized_representation:
            kwargs['result_map'] = {bson.ObjectId(trial_id): {bson.ObjectId(bench_id): res_id
                                                              for bench_id, res_id in inner_map.items()}
                                    for trial_id, inner_map in serialized_representation['result_map'].items()}
        return super().deserialize(serialized_representation, db_client, **kwargs)

    def _set_property(self, serialized_key: str, new_value: typing.Union[str, dict, list, int, float, bson.ObjectId]):
        """
        Helper to track updates to a single property in the serialized object
        :param serialized_key: The name of the property in the serialized form
        :param new_value: The new value of hte property
        :return: void
        """
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        self._updates['$set'][serialized_key] = new_value

    def _add_to_list(self, serialized_key: str,
                     new_elements: typing.List[typing.Union[str, dict, list, int, float, bson.ObjectId]]):
        """
        Helper to append new elements to a list key
        :param serialized_key: The serialized key
        :param new_elements: A collection of new elements to add to the end of the list
        :return:
        """
        if not isinstance(new_elements, list):
            if isinstance(new_elements, collections.Iterable):
                new_elements = list(new_elements)
            else:
                new_elements = [new_elements]
        if len(new_elements) > 0:
            if '$push' not in self._updates:
                self._updates['$push'] = {}
            if serialized_key in self._updates['$push']:
                self._updates['$push'][serialized_key]['$each'] += new_elements
            else:
                self._updates['$push'][serialized_key] = {'$each': new_elements}

    def _add_to_set(self, serialized_key: str,
                    new_elements: typing.List[typing.Union[str, dict, list, int, float, bson.ObjectId]]):
        """
        Helper to collect changes to a set property on the object,
        such as the set of system ids to be tested, or the set of image sources to use.
        Do not call this externally, but child classes should call this when any of their collections
        are changed to accumulate changes to be saved with 'save_updates'

        Internally uses mongod addToSet, so duplicate entries will be omitted.

        :param serialized_key: The key the set is stored as in the database, using dot notation
        :param new_elements: New elements to be stored in the set
        :return: void
        """
        if not isinstance(new_elements, set):
            if isinstance(new_elements, collections.Iterable):
                new_elements = set(new_elements)
            else:
                new_elements = {new_elements}
        if len(new_elements) > 0:
            if '$addToSet' not in self._updates:
                self._updates['$addToSet'] = {}
            existing = (set(self._updates['$addToSet'][serialized_key]['$each'])
                        if serialized_key in self._updates['$addToSet'] else set())
            self._updates['$addToSet'][serialized_key] = {'$each': list(new_elements | existing)}
