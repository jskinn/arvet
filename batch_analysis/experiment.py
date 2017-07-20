import enum
import bson.objectid
import database.entity


class ProgressState(enum.Enum):
    """
    Enum for tracking the progress of different trials and benchmarks in the experiment.
    Mostly used internally, aside from tests and interpreting db structure,
    you shouldn't need this.
    """
    UNSTARTED = 0
    RUNNING = 1
    FINISHED = 2


class Experiment(database.entity.Entity):

    def __init__(self, trainers=None, trainees=None, image_sources=None, systems=None, benchmarks=None,
                 trial_results=None, benchmark_results=None, training_map=None, trial_map=None, benchmark_map=None,
                 id_=None):
        super().__init__(id_=id_)
        self._trainers = set(trainers) if trainers is not None else set()
        self._trainees = set(trainees) if trainees is not None else set()
        self._image_sources = set(image_sources) if image_sources is not None else set()
        self._systems = set(systems) if systems is not None else set()
        self._benchmarks = set(benchmarks) if benchmarks is not None else set()
        self._trial_results = set(trial_results) if trial_results is not None else set()
        self._benchmark_results = set(benchmark_results) if benchmark_results is not None else set()

        self._training_map = dict(training_map) if training_map is not None else {}
        self._trial_map = dict(trial_map) if trial_map is not None else {}
        self._benchmark_map = dict(benchmark_map) if benchmark_map is not None else {}

        self._updates = {}

    def do_imports(self, db_client, save_changes=True):
        """
        Perform imports for this experiment.
        By default, this calls import_systems, import_image_sources, and import_benchmarks.
        If you need to import extra things, override this.

        This needs to be sure that it won't create duplicate objects,
        it will be called repeatedly by the scheduler.

        :param db_client: The database client, we need it for importing, and for saving changes
        :param save_changes: Whether we should save the changes when done, since we always need the db_client
        :return: void
        """
        new_trainers = set(self.import_trainers(db_client))
        new_trainees = set(self.import_trainees(db_client))
        new_systems = set(self.import_systems(db_client))
        new_image_sources = set(self.import_image_sources(db_client))
        new_benchmarks = set(self.import_benchmarks(db_client))

        self._add_to_set('trainers', new_trainers - self._trainers)
        self._add_to_set('trainees', new_trainees - self._trainees)
        self._add_to_set('systems', new_systems - self._systems)
        self._add_to_set('image_sources', new_image_sources - self._image_sources)
        self._add_to_set('benchmarks', new_benchmarks - self._benchmarks)
        self._trainers = self._trainers | new_trainers
        self._trainees = self._trainees | new_trainees
        self._systems = self._systems | new_systems
        self._image_sources = self._image_sources | new_image_sources
        self._benchmarks = self._benchmarks | new_benchmarks

        if save_changes:
            self.save_updates(db_client)

    def schedule_tasks(self, job_system, db_client=None):
        """
        Import new systems and schedule new tasks using the job systems
        :param job_system: The job system used to schedule new tasks
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return: void
        """

        # Update training map for new/missing trainers and trainees
        for trainer_id in self._trainers:
            if trainer_id not in self._training_map:
                self._training_map[trainer_id] = {}
            for trainee_id in self._trainees:
                if trainee_id not in self._training_map[trainer_id]:
                    self._change_training_state(trainer_id, trainee_id, ProgressState.UNSTARTED)

        # Update trials map for new/missing image sources and systems
        for system_id in self._systems:
            if system_id not in self._trial_map:
                self._trial_map[system_id] = {}
            for image_source_id in self._image_sources:
                if image_source_id not in self._trial_map[system_id]:
                    self._change_trial_state(system_id, image_source_id, ProgressState.UNSTARTED)

        # Update benchmarks map for new/missing trial results and benchmarks
        for trial_result_id in self._trial_results:
            if trial_result_id not in self._benchmark_map:
                self._benchmark_map[trial_result_id] = {}
            for benchmark_id in self._benchmarks:
                if benchmark_id not in self._benchmark_map[trial_result_id]:
                    self._change_result_state(trial_result_id, benchmark_id, ProgressState.UNSTARTED)

        # Schedule new training using the job system
        for trainer_id in self._training_map.keys():
            for trainee_id in self._training_map[trainer_id].keys():
                if self._training_map[trainer_id][trainee_id] == ProgressState.UNSTARTED:
                    job_system.queue_train_system(trainer_id, trainee_id, self.identifier)
                    self._change_training_state(trainer_id, trainee_id, ProgressState.RUNNING)

        # Schedule new trials using the job system
        for system_id in self._trial_map.keys():
            for image_source_id in self._trial_map[system_id].keys():
                if self._trial_map[system_id][image_source_id] == ProgressState.UNSTARTED:
                    job_system.queue_run_system(system_id, image_source_id, self.identifier)
                    self._change_trial_state(system_id, image_source_id, ProgressState.RUNNING)

        # Schedule new benchmarks using the job system
        for trial_result_id in self._benchmark_map.keys():
            for benchmark_id in self._benchmark_map[trial_result_id].keys():
                if self._benchmark_map[trial_result_id][benchmark_id] == ProgressState.UNSTARTED:
                    job_system.queue_benchmark_result(trial_result_id, benchmark_id, self.identifier)
                    self._change_result_state(trial_result_id, benchmark_id, ProgressState.RUNNING)

        if db_client is not None:
            self.save_updates(db_client)

    def retry_training(self, trainer_id, trainee_id, db_client=None):
        """
        Training has failed for some reason, and we should retry it.
        Change the state in the database to unstarted again so that a new run will be scheduled
        the next time schedule_tasks is called.
        :param trainer_id: The id of the trainer
        :param trainee_id: The id of the trainee being trained
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return:
        """
        if (trainer_id in self._training_map and trainee_id in self._training_map[trainer_id] and
                self._training_map[trainer_id][trainee_id] == ProgressState.RUNNING):
            self._change_training_state(trainer_id, trainee_id, ProgressState.UNSTARTED)
            if db_client is not None:
                self.save_updates(db_client)

    def add_system(self, trainer_id, trainee_id, system_id, db_client=None):
        """
        Training has completed, producing a new system.
        Update the map so we don't schedule it again, and store the new system for testing
        :param trainer_id: The id of the trainer that did the training
        :param trainee_id: The id if the trainee that was trained
        :param system_id: The id of the newly created system
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return:
        """
        if trainer_id in self._trainers and trainee_id in self._trainees:
            self._change_training_state(trainer_id, trainee_id, ProgressState.FINISHED)
            self._systems.add(system_id)
            self._trial_map[system_id] = {}
            for image_source_id in self._image_sources:
                self._change_trial_state(system_id, image_source_id, ProgressState.UNSTARTED)
            self._add_to_set('systems', {system_id})
            if db_client is not None:
                self.save_updates(db_client)

    def retry_trial(self, system_id, image_source_id, db_client=None):
        """
        A given trial has failed for some reason, and we should retry it.
        Change the state in the database to unstarted again so that a new run will be scheduled
        the next time schedule_tasks is called.
        :param system_id: The id of the system under test
        :param image_source_id: The id of the image source used for testing
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return:
        """
        if (system_id in self._trial_map and image_source_id in self._trial_map[system_id] and
                self._trial_map[system_id][image_source_id] == ProgressState.RUNNING):
            self._change_trial_state(system_id, image_source_id, ProgressState.UNSTARTED)
            if db_client is not None:
                self.save_updates(db_client)

    def add_trial_result(self, system_id, image_source_id, trial_result_id, db_client=None):
        """
        A trial has completed, producing a new trial result.
        Update the trial state so we don't schedule it again,
        and store the new id for benchmarking
        :param system_id: The id of the system under test
        :param image_source_id: The id if the image source used for testing
        :param trial_result_id: The id of the newly created trial result
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return:
        """
        if system_id in self._systems and image_source_id in self._image_sources:
            self._change_trial_state(system_id, image_source_id, ProgressState.FINISHED)
            self._trial_results.add(trial_result_id)
            self._benchmark_map[trial_result_id] = {}
            for benchmark_id in self._benchmarks:
                self._change_result_state(trial_result_id, benchmark_id, ProgressState.UNSTARTED)
            self._add_to_set('trial_results', {trial_result_id})
            if db_client is not None:
                self.save_updates(db_client)

    def retry_benchmark(self, trial_result_id, benchmark_id, db_client=None):
        """
        A given benchmark has failed for some reason, and we should retry it.
        Change the state in the database to unstarted again so that a new run will be scheduled
        the next time schedule_tasks is called.
        :param trial_result_id: The id of the trial result being measured
        :param benchmark_id: The id of the benchmark to use
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return:
        """
        if (trial_result_id in self._benchmark_map and benchmark_id in self._benchmark_map[trial_result_id] and
                self._benchmark_map[trial_result_id][benchmark_id] == ProgressState.RUNNING):
            self._change_result_state(trial_result_id, benchmark_id, ProgressState.UNSTARTED)
            if db_client is not None:
                self.save_updates(db_client)

    def add_benchmark_result(self, trial_result_id, benchmark_id, benchmark_result_id, db_client=None):
        """
        A benchmark has completed, producing a new benchmark result.
        Update the trial state so we don't schedule it again,
        and store the result id for display and analysis
        :param trial_result_id: The id of the trial result being measured
        :param benchmark_id: The id if the benchmark used to measure the result
        :param benchmark_result_id: The id of the newly created benchmark result
        :param db_client: The database client if we want to save changes immediately, None otherwise.
        :return:
        """
        if trial_result_id in self._trial_results and benchmark_id in self._benchmarks:
            self._change_result_state(trial_result_id, benchmark_id, ProgressState.FINISHED)
            self._benchmark_results.add(benchmark_result_id)
            self._add_to_set('benchmark_results', {benchmark_result_id})
            if db_client is not None:
                self.save_updates(db_client)

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

    def import_trainers(self, db_client):
        """
        Import trainers to train new systems for the experiment.
        Should return the database ids of the newly created trainers.
        This may involve importing additional training datasets,
        which should be stored along with the settings in the trainer
        :param db_client: The database client
        :return: A set of database ids for system trainers
        """
        return set()

    def import_trainees(self, db_client):
        """
        Create or import trainees into the database for this experiment.
        These determine which systems will be trained, and some of the settings for doing so.
        Note that trainees and trainers are different, see core.trained_system for each of the types.
        In genral, a trainer holds some image data, and a process for feeding it to a trainee,
        which works like a builder to create a new trained system.
        :param db_client: The database client
        :return: The set of ids of new trainees. May include existing ids.
        """
        return set()

    def import_image_sources(self, db_client):
        """
        Import the datasets and other image sources associated with this experiment.
        Should return the database ids of the image sources, this may include any image sources
        that have already been imported (we use sets to remove duplicates)
        :param db_client: The database client, used to do the importing
        :return: A collection of the imported image source ids. May include existing ids.
        """
        return set()

    def import_systems(self, db_client):
        """
        Import vision systems associated with this experiment.
        Must return the database ids of the imported systems, which may include existing system ids
        :param db_client: The database client used to do the importing
        :return: A collection of the database ids of the imported image sources
        """
        return set()

    def import_benchmarks(self, db_client):
        """
        Import benchmarks associated with this
        :param db_client:
        :return:
        """
        return set()

    def _change_training_state(self, trainer_id, trainee_id, state):
        """
        Helper to change a value in the training map,
        accumulating a change query for later saving
        :param trainer_id: The id of the trainer doing the training
        :param trainee_id: The id of the trainee being trained
        :param state: The new state value
        :return:
        """
        self._training_map[trainer_id][trainee_id] = state
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        s_key = 'training_map.{0}.{1}'.format(str(trainer_id), str(trainee_id))
        self._updates['$set'][s_key] = int(state.value)

    def _change_trial_state(self, system_id, image_source_id, state):
        """
        Helper to change a value in the trial map,
        accumulating a change query for later saving
        :param system_id: The id of the system
        :param image_source_id: The id of the image source
        :param state: The new state value
        :return:
        """
        self._trial_map[system_id][image_source_id] = state
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        s_key = 'trial_map.{0}.{1}'.format(str(system_id), str(image_source_id))
        self._updates['$set'][s_key] = int(state.value)

    def _change_result_state(self, trial_result_id, benchmark_id, state):
        """
        Change a value in the benchmark map,
        accumulating changes for later saving
        :param trial_result_id:
        :param benchmark_id:
        :param state:
        :return:
        """
        self._benchmark_map[trial_result_id][benchmark_id] = state
        if '$set' not in self._updates:
            self._updates['$set'] = {}
        s_key = 'benchmark_map.{0}.{1}'.format(str(trial_result_id), str(benchmark_id))
        self._updates['$set'][s_key] = int(state.value)

    def _add_to_set(self, serialized_key, new_elements):
        """
        Collect changes to a set property on the object,
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
            existing = (set(self._updates['$addToSet'][serialized_key])
                        if serialized_key in self._updates['$addToSet'] else set())
            self._updates['$addToSet'][serialized_key] = {'$each': list(set(new_elements) | existing)}

    def validate(self):
        return super().validate()

    def serialize(self):
        serialized = super().serialize()
        serialized['trainers'] = [str(oid) for oid in self._trainers]
        serialized['trainees'] = [str(oid) for oid in self._trainees]
        serialized['image_sources'] = [str(oid) for oid in self._image_sources]
        serialized['systems'] = [str(oid) for oid in self._systems]
        serialized['trial_results'] = [str(oid) for oid in self._trial_results]
        serialized['benchmarks'] = [str(oid) for oid in self._benchmarks]
        serialized['benchmark_results'] = [str(oid) for oid in self._benchmark_results]
        # Note: The way these serialize based on the sets above keeps their keys in sync
        serialized['training_map'] = {
            str(trainer_id): {
                str(trainee_id): (int(self._training_map[trainer_id][trainee_id].value)
                                  if trainer_id in self._training_map and
                                  trainee_id in self._training_map[trainer_id]
                                  else int(ProgressState.UNSTARTED.value))
                for trainee_id in self._trainees
            } for trainer_id in self._trainers
        }
        serialized['trial_map'] = {
            str(system_id): {
                str(source_id): (int(self._trial_map[system_id][source_id].value)
                                 if system_id in self._trial_map and
                                 source_id in self._trial_map[system_id] else int(ProgressState.UNSTARTED.value))
                for source_id in self._image_sources
            } for system_id in self._systems
        }
        serialized['benchmark_map'] = {
            str(trial_id): {
                str(benchmark_id): (int(self._benchmark_map[trial_id][benchmark_id].value)
                                    if trial_id in self._benchmark_map and
                                    benchmark_id in self._benchmark_map[trial_id] else ProgressState.UNSTARTED)
                for benchmark_id in self._benchmarks
            } for trial_id in self._trial_results
        }
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trainers' in serialized_representation:
            kwargs['trainers'] = set(bson.objectid.ObjectId(oid) for oid in serialized_representation['trainers'])
        if 'trainees' in serialized_representation:
            kwargs['trainees'] = set(bson.objectid.ObjectId(oid) for oid in serialized_representation['trainees'])
        if 'image_sources' in serialized_representation:
            kwargs['image_sources'] = set(bson.objectid.ObjectId(oid)
                                          for oid in serialized_representation['image_sources'])
        if 'systems' in serialized_representation:
            kwargs['systems'] = set(bson.objectid.ObjectId(oid) for oid in serialized_representation['systems'])
        if 'benchmarks' in serialized_representation:
            kwargs['benchmarks'] = set(bson.objectid.ObjectId(oid) for oid in serialized_representation['benchmarks'])
        if 'trial_results' in serialized_representation:
            kwargs['trial_results'] = set(bson.objectid.ObjectId(oid)
                                          for oid in serialized_representation['trial_results'])
        if 'benchmark_results' in serialized_representation:
            kwargs['benchmark_results'] = set(bson.objectid.ObjectId(oid)
                                              for oid in serialized_representation['benchmark_results'])
        # Rebuild the training map, trial map, and benchmark map
        # This has the advantage of removing unassociated keys, and adding missing keys
        kwargs['training_map'] = {
            trainer_id: {
                trainee_id: ProgressState(serialized_representation['training_map'][str(trainer_id)][str(trainee_id)])
                if 'training_map' in serialized_representation and
                   str(trainer_id) in serialized_representation['training_map'] and
                   str(trainee_id) in serialized_representation['training_map'][str(trainer_id)]
                else ProgressState.UNSTARTED
                for trainee_id in kwargs['trainees']
            } for trainer_id in kwargs['trainers']
        }
        kwargs['trial_map'] = {
            system_id: {
                source_id: ProgressState(serialized_representation['trial_map'][str(system_id)][str(source_id)])
                if 'trial_map' in serialized_representation and
                   str(system_id) in serialized_representation['trial_map'] and
                   str(source_id) in serialized_representation['trial_map'][str(system_id)]
                else ProgressState.UNSTARTED
                for source_id in kwargs['image_sources']
            } for system_id in kwargs['systems']
        }
        kwargs['benchmark_map'] = {
            trial_id: {
                benchmark_id: ProgressState(
                    serialized_representation['benchmark_map'][str(trial_id)][str(benchmark_id)])
                if 'benchmark_map' in serialized_representation and
                   str(trial_id) in serialized_representation['benchmark_map'] and
                   str(benchmark_id) in serialized_representation['benchmark_map'][str(trial_id)]
                else ProgressState.UNSTARTED
                for benchmark_id in kwargs['benchmarks']
            } for trial_id in kwargs['trial_results']
        }
        return super().deserialize(serialized_representation, db_client, **kwargs)
