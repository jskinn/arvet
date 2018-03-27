# Copyright (c) 2017, John Skinner
import abc
import os.path
import collections
import typing
import bson
import arvet.batch_analysis.task_manager
import arvet.batch_analysis.invalidate
import arvet.config.path_manager
import arvet.database.client
import arvet.database.entity
import arvet.database.entity_registry as entity_registry
import arvet.util.database_helpers as dh


class Experiment(arvet.database.entity.Entity, metaclass=arvet.database.entity.AbstractEntityMetaclass):
    """
    A model for an experiment. The role of the experiment is to decide which systems should be run with
    which datasets, and which benchmarks should be used to measure them.
    They form a collection of results for us to analyse and write papers from

    Fundamentally, an experiment is a bunch of groups of ids for core object types,
    which will be mixed and matched in trials and benchmarks to produce our desired results.
    The groupings of the ids are meaningful for the particular experiment, to convey some level
    of association or meta-data that we can't articulate or encapsulate in the image_metadata.
    """

    def __init__(self, trial_map: dict = None, enabled: bool = True,
                 id_: typing.Union[bson.ObjectId, None] = None):
        super().__init__(id_=id_)
        self.enabled = enabled
        self._trial_map = trial_map if trial_map is not None else {}
        self._updates = {}

    @abc.abstractmethod
    def do_imports(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                   path_manager: arvet.config.path_manager.PathManager,
                   db_client: arvet.database.client.DatabaseClient):
        """
        Perform imports for this experiment,
        This is where object creation should be performed (like creating benchmarks or untrained systems)
        We should also use the task manager to create import tasks, but not any other kinds of tasks.

        :param task_manager: The task manager, for creating import tasks (import datasets, generate datasets)
        :param path_manager: A path manager for locating files and folders on disk
        :param db_client: The database client, to save declared objects that are too small for an import
        :return: void
        """
        pass

    @abc.abstractmethod
    def schedule_tasks(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                       db_client: arvet.database.client.DatabaseClient):
        """
        This is where you schedule the core tasks to train a system, run a system, or perform a benchmark,
        :param task_manager: The the task manager used to create tasks
        :param db_client: The database client, so that we can deserialize entities and check compatibility
        :return: void
        """
        pass

    def save_updates(self, db_client: arvet.database.client.DatabaseClient):
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

    def plot_results(self, db_client: arvet.database.client.DatabaseClient):
        """
        Visualise the results from this experiment.
        Non-compulsory, but will be called from plot_results.py
        :param db_client:
        :return:
        """
        pass

    def export_data(self, db_client: arvet.database.client.DatabaseClient):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to. Naturally, this is optional.
        :param db_client:
        :return:
        """
        pass

    def perform_analysis(self, db_client: arvet.database.client.DatabaseClient):
        """
        Method by which an experiment can perform large-scale analysis as a job.
        This is what is called by AnalyseResultsTask
        This should save it's output somehow, preferably within the output folder given by get_output_folder()
        :param db_client: The database client, for loading trials and benchmark results.
        :return:
        """
        pass

    def schedule_all(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                     db_client: arvet.database.client.DatabaseClient,
                     systems: typing.List[bson.ObjectId],
                     image_sources: typing.List[bson.ObjectId],
                     benchmarks: typing.List[bson.ObjectId],
                     repeats: int = 1,
                     allow_incomplete_benchmarks: bool = False):
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
        :param repeats: The number of times to repeat
        :param allow_incomplete_benchmarks: Whether to run benchmarks when not all the trials have completed yet.
        :return: void
        """
        # Trial results will be collected as we go
        trial_results_to_benchmark = []
        repeats = max(repeats, 1)   # always at least 1 repeat
        changes = 0
        anticipated_changes = 0

        # For each image dataset, run libviso with that dataset, and store the result in the trial map
        for image_source_id in image_sources:
            image_source = dh.load_object(db_client, db_client.image_source_collection, image_source_id)
            if image_source is None:
                continue
            for system_id in systems:
                system = dh.load_object(db_client, db_client.system_collection, system_id)
                if system is not None and system.is_image_source_appropriate(image_source):
                    trial_result_group = set()
                    for repeat in range(repeats):
                        task = task_manager.get_run_system_task(
                            system_id=system.identifier,
                            image_source_id=image_source.identifier,
                            repeat=repeat,
                            expected_duration='8:00:00',
                            memory_requirements='12GB'
                        )
                        if not task.is_finished:
                            task_manager.do_task(task)
                            anticipated_changes += 1
                        else:
                            trial_result_group.add(task.result)

                    if self.store_trial_results(system_id, image_source_id, trial_result_group, db_client):
                        changes += 1

                    if len(trial_result_group) >= repeats or allow_incomplete_benchmarks:
                        # Schedule benchmarks for the group
                        trial_results_to_benchmark.append((system_id, image_source_id, trial_result_group))

        # Benchmark trial results collected in the previous step
        for benchmark_id in benchmarks:
            benchmark = dh.load_object(db_client, db_client.benchmarks_collection, benchmark_id)
            if benchmark is not None:
                for system_id, image_source_id, trial_result_group in trial_results_to_benchmark:
                    is_appropriate = True
                    for trial_result_id in trial_result_group:
                        trial_result = dh.load_object(db_client, db_client.trials_collection, trial_result_id)
                        if trial_result is None or not benchmark.is_trial_appropriate(trial_result):
                            is_appropriate = False
                            break
                    if is_appropriate:
                        task = task_manager.get_benchmark_task(
                            trial_result_ids=trial_result_group,
                            benchmark_id=benchmark.identifier,
                            expected_duration='6:00:00',
                            memory_requirements='6GB'
                        )
                        if not task.is_finished:
                            task_manager.do_task(task)
                            anticipated_changes += 1
                        else:
                            if self.store_benchmark_result(system_id, image_source_id, benchmark_id, task.result):
                                changes += 1
        return changes, anticipated_changes

    def store_trial_results(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId,
                            trial_result_ids: typing.Iterable[bson.ObjectId],
                            db_client: arvet.database.client.DatabaseClient) -> bool:
        """
        Store the results of running a particular system with a particular image source,
        that is, a group of trial results against the system and image source that produced them.
        Call this if you schedule trials yourself, it will be called automatically as part of schedule_all
        :param system_id: The system id that peformed these trials
        :param image_source_id: The image source id given to the system to produce these trials
        :param trial_result_ids: An iterable of trial result ids, allowing multiple repeats
        :param db_client: The database client, to remove any existing benchmark results
        :return: true iff we stored a change, false if we did not
        """
        if system_id not in self._trial_map:
            self._trial_map[system_id] = {}
        if image_source_id not in self._trial_map[system_id]:
            self._trial_map[system_id][image_source_id] = {'trials': list(trial_result_ids), 'results': {}}
            self._set_property('trial_map.{0}.{1}'.format(system_id, image_source_id),
                               serialize_trial_obj(self._trial_map[system_id][image_source_id]))
            return True
        elif set(self._trial_map[system_id][image_source_id]['trials']) != set(trial_result_ids):
            # Set of trials has changed, invalidate all benchmark results and reset
            for benchmark_result_id in self._trial_map[system_id][image_source_id]['results'].items():
                arvet.batch_analysis.invalidate.invalidate_benchmark_result(db_client, benchmark_result_id)

            # Reset the trial map with the new trial set
            self._trial_map[system_id][image_source_id] = {'trials': list(trial_result_ids), 'results': {}}
            self._set_property('trial_map.{0}.{1}'.format(system_id, image_source_id),
                               serialize_trial_obj(self._trial_map[system_id][image_source_id]))
            return True
        return False

    def get_trial_results(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId) \
            -> typing.Set[bson.ObjectId]:
        """
        Get the trial result produced by running a given system with a given image source.
        Return None if the system has not been run with that image source.
        :param system_id: The id of the system
        :param image_source_id: The id of the image source
        :return: The id of the trial result, or None if the trial has not been performed
        """
        if system_id in self._trial_map and image_source_id in self._trial_map[system_id]:
            return self._trial_map[system_id][image_source_id]['trials']
        return set()

    def store_benchmark_result(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId,
                               benchmark_id: bson.ObjectId, benchmark_result_id: bson.ObjectId) -> bool:
        """
        Store the result of measuring a particular set of trials with a particular benchmark.
        Cannot store results for trials that are not stored in this experiment, call 'store_trial_results' first.
        :param system_id: The id of the system used to produce the benchmarked trials
        :param image_source_id: The id of the image source to perform the benchmarked trials
        :param benchmark_id: The id of the benchmark used
        :param benchmark_result_id: The id of the benchmark result
        :return: true if a change was made to the trial map
        """
        if system_id in self._trial_map and image_source_id in self._trial_map[system_id]:
            self._trial_map[system_id][image_source_id]['results'][benchmark_id] = benchmark_result_id
            self._set_property('trial_map.{0}.{1}'.format(system_id, image_source_id),
                               serialize_trial_obj(self._trial_map[system_id][image_source_id]))
            return True
        return False

    def get_benchmark_result(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId,
                             benchmark_id: bson.ObjectId) -> typing.Union[bson.ObjectId, None]:
        """
        Get the results of benchmarking the results of a particular system on a particular image source,
        using the given benchmark
        :param system_id: The id of the system used to produce the benchmarked trials
        :param image_source_id: The id of the image source to perform the benchmarked trials
        :param benchmark_id: The id of the benchmark used
        :return: The id of the result object, or None if the trials have not been measured.
        """
        if system_id in self._trial_map and \
                image_source_id in self._trial_map[system_id] and \
                benchmark_id in self._trial_map[system_id][image_source_id]['results']:
            return self._trial_map[system_id][image_source_id]['results'][benchmark_id]
        return None

    def serialize(self):
        """
        Serialize the experiment for storage in the database
        :return:
        """
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'arvet:batch_analysis:experiment:Experiment', 1)
        serialized['enabled'] = self.enabled
        serialized['trial_map'] = {
            str(sys_id): {
                str(source_id): serialize_trial_obj(trial_obj)
                for source_id, trial_obj in inner_map.items()
            }
            for sys_id, inner_map in self._trial_map.items()
        }
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'enabled' in serialized_representation:
            kwargs['enabled'] = bool(serialized_representation['enabled'])
        if 'trial_map' in serialized_representation:
            kwargs['trial_map'] = deserialize_trial_map(serialized_representation['trial_map'], db_client)
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_output_folder(cls):
        """
        Get a unique output folder for this experiment.
        Really, we just name the folder after the experiment, but it's nice to do this in a standardized way.
        :return:
        """
        return os.path.join('results', cls.__name__)

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


def serialize_trial_obj(trial_obj: dict):
    """
    Convert an entry in the trial map to a format that can be saved in the database.
    This makes sure all our keys are strings, and the
    :param trial_obj:
    :return:
    """
    return {
        'trials': list(trial_obj['trials']),
        'results': {
            str(benchmark_id): result_id
            for benchmark_id, result_id in trial_obj['results'].items()
        }
    }


def deserialize_trial_map(s_trial_map: dict, db_client: arvet.database.client.DatabaseClient) -> dict:
    """
    A helper to deserialize the trial map, checking and removing invalid ids as we go
    :param s_trial_map: The serialized trial map
    :param db_client:
    :return:
    """
    trial_map = {
        bson.ObjectId(sys_id): {
            bson.ObjectId(source_id): {
                'trials': set(trial_obj['trials']),
                'results': {
                    bson.ObjectId(benchmark_id): result_id
                    for benchmark_id, result_id in trial_obj['results'].items()
                }
            }
            for source_id, trial_obj in inner_map.items()
        }
        for sys_id, inner_map in s_trial_map.items()
    }

    # Delete invalid systems
    invalid_system_ids = dh.check_many_references(db_client.system_collection, trial_map.keys())
    for system_id in invalid_system_ids:
        del trial_map[system_id]

    # Delete invalid image sources
    invalid_image_source_ids = dh.check_many_references(db_client.image_source_collection, set(
        image_source_id
        for image_source_map in trial_map.values()
        for image_source_id in image_source_map.keys()
    ))
    for image_source_id in invalid_image_source_ids:
        for image_source_map in trial_map.values():
            if image_source_id in image_source_map:
                del image_source_map[image_source_id]

    # Delete invalid trial results
    invalid_trial_result_ids = dh.check_many_references(db_client.trials_collection, set(
        trial_result_id
        for image_source_map in trial_map.values()
        for trials_obj in image_source_map.values()
        for trial_result_id in trials_obj['trials']
    ))
    if len(invalid_trial_result_ids) > 0:
        for image_source_map in trial_map.values():
            to_delete = set()
            for image_source_id, trials_obj in image_source_map.items():
                remaining = set(trials_obj['trials']) - invalid_trial_result_ids
                if len(remaining) <= 0:
                    # No trials for this image source are valid, delete the whole entry
                    to_delete.add(image_source_id)
                else:
                    # There's some number of valid trial result ids in this group, shrink to that many
                    trials_obj['trials'] = remaining
            for image_source_id in to_delete:
                del image_source_map[image_source_id]

    # Delete results for invalid benchmarks
    invalid_benchmark_ids = dh.check_many_references(db_client.benchmarks_collection, set(
        benchmark_id
        for image_source_map in trial_map.values()
        for trials_obj in image_source_map.values()
        for benchmark_id in trials_obj['results'].keys()
    ))
    for benchmark_id in invalid_benchmark_ids:
        for image_source_map in trial_map.values():
            for trials_obj in image_source_map.values():
                if benchmark_id in trials_obj['results']:
                    del trials_obj['results'][benchmark_id]

    # Delete invalid benchmark results
    invalid_benchmark_result_ids = dh.check_many_references(db_client.results_collection, set(
        benchmark_result_id
        for image_source_map in trial_map.values()
        for trials_obj in image_source_map.values()
        for benchmark_result_id in trials_obj['results'].values()
    ))
    for invalid_benchmark_result_id in invalid_benchmark_result_ids:
        for image_source_map in trial_map.values():
            for trials_obj in image_source_map.values():
                for benchmark_id in set(
                    benchmark_id
                    for benchmark_id, benchmark_result_id in trials_obj['results'].items()
                    if benchmark_result_id == invalid_benchmark_result_id
                ):
                    del trials_obj['results'][benchmark_id]

    return trial_map


def create_experiment(db_client: arvet.database.client.DatabaseClient,
                      experiment_type: typing.Type[Experiment]) -> None:
    """
    Store an experiment in the database. Experiments are uniquely identified by type,
    so it will not store it if another experiment of that type already exists.
    :param db_client: The database client
    :param experiment_type: The experiment type to create
    :return: void
    """
    if db_client.experiments_collection.find({
            '_type': entity_registry.get_type_name(experiment_type)}).count() <= 0:
        db_client.experiments_collection.insert_one(experiment_type().serialize())
