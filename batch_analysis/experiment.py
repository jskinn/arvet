import abc
import database.entity


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

    def __init__(self, id_=None):
        super().__init__(id_=id_)
        self._updates = {}

    @abc.abstractmethod
    def do_imports(self, task_manager, db_client):
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
    def schedule_tasks(self, task_manager, db_client):
        """
        This is where you schedule the core tasks to train a system, run a system, or perform a benchmark,
        :param task_manager: The the task manager used to create tasks
        :param db_client: The database client, so that we can deserialize entities and check compatibility
        :return: void
        """
        pass

    def save_updates(self, db_client):
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

    def plot_results(self, db_client):
        """
        Visualise the results from this experiment
        :param db_client:
        :return:
        """
        pass

    def _set_property(self, serialized_key, new_value):
        """
        Helper to track updates to a single property in the serialized object
        :param serialized_key: The name of the property in the serialized form
        :param new_value: The new value of hte property
        :return: void
        """
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        self._updates['$set'][serialized_key] = new_value

    def _add_to_list(self, serialized_key, new_elements):
        """
        Helper to append new elements to a list key
        :param serialized_key: The serialized key
        :param new_elements: A collection of new elements to add to the end of the list
        :return:
        """
        if len(new_elements) > 0:
            if '$push' not in self._updates:
                self._updates['$push'] = {}
            if serialized_key in self._updates['$push']:
                self._updates['$push'][serialized_key]['$each'] += list(new_elements)
            else:
                self._updates['$push'][serialized_key] = {'$each': list(new_elements)}

    def _add_to_set(self, serialized_key, new_elements):
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
        if len(new_elements) > 0:
            if '$addToSet' not in self._updates:
                self._updates['$addToSet'] = {}
            existing = (set(self._updates['$addToSet'][serialized_key]['$each'])
                        if serialized_key in self._updates['$addToSet'] else set())
            self._updates['$addToSet'][serialized_key] = {'$each': list(set(new_elements) | existing)}
