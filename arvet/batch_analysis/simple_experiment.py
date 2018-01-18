import typing
import bson
import arvet.util.database_helpers as dh
import arvet.core.system
import arvet.core.benchmark
import arvet.config.path_manager
import arvet.database.client
import arvet.database.entity
import arvet.batch_analysis.experiment
import arvet.batch_analysis.task_manager


class SimpleExperiment(arvet.batch_analysis.experiment.Experiment,
                       metaclass=arvet.database.entity.AbstractEntityMetaclass):
    """
    A subtype of experiment for simple cases where all systems are run with all datasets,
    and then are run with all benchmarks.
    If this common case is not your experiment, override base Experiment instead
    """

    def __init__(self, systems=None,
                 datasets=None,
                 benchmarks=None,
                 repeats=1,
                 trial_map=None, result_map=None, enabled=True, id_=None):
        """
        Constructor. We need parameters to load the different stored parts of this experiment
        :param systems: The map of systems in this experiment
        :param datasets: The map of datasets in this experiment
        :param benchmarks: The map of benchmarks in this experiment
        :param repeats: The number of repeats for the systems
        :param trial_map: The map of systems and image sources to trials
        :param result_map: The map of trials and benchmarks to results
        :param enabled: Is this experiment enabled
        :param id_: The id of the experiment, passed to Entity constructor
        """
        super().__init__(id_=id_, trial_map=trial_map, result_map=result_map, enabled=enabled)
        self._systems = systems if systems is not None else {}
        self._datasets = datasets if datasets is not None else {}
        self._benchmarks = benchmarks if benchmarks is not None else {}
        self._repeats = int(repeats)

    @property
    def systems(self) -> typing.Mapping[str, bson.ObjectId]:
        return self._systems

    @property
    def datasets(self) -> typing.Mapping[str, bson.ObjectId]:
        return self._datasets

    @property
    def benchmarks(self) -> typing.Mapping[str, bson.ObjectId]:
        return self._benchmarks

    def schedule_tasks(self, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                       db_client: arvet.database.client.DatabaseClient):
        """
        Schedule the running of systems with image sources, and the benchmarking of the trial results so produced.
        :param task_manager:
        :param db_client:
        :return:
        """
        # Schedule all combinations of systems with the generated datasets
        self.schedule_all(task_manager=task_manager,
                          db_client=db_client,
                          systems=list(self._systems.values()),
                          image_sources=list(self._datasets.values()),
                          benchmarks=list(self._benchmarks.values()),
                          repeats=self._repeats)

    def import_system(self, name: str, system: arvet.core.system.VisionSystem,
                      db_client: arvet.database.client.DatabaseClient) -> None:
        """
        Import a system into the experiment. It will be run with all the image sources.
        :param name: The name of the system
        :param system: The system object, to serialize and save if necessary
        :param db_client: The database client, to use to save the system
        :return:
        """
        if name not in self._systems:
            self._systems[name] = dh.add_unique(db_client.system_collection, system)
            self._set_property('systems.{0}'.format(name), self._systems[name])

    def import_dataset(self, name: str, task_manager: arvet.batch_analysis.task_manager.TaskManager,
                       path_manager: arvet.config.path_manager.PathManager,
                       module_name: str, path: str, additional_args: dict = None,
                       num_cpus: int = 1, num_gpus: int = 0,
                       memory_requirements: str = '3GB', expected_duration: str = '12:00:00') -> None:
        """
        Import a dataset at a given path, using a given module.
        Has all the arguments of get_import_dataset_task, which are passed through
        :param name: The name to store the dataset as
        :param task_manager: The task manager, for scheduling
        :param path_manager: The path manager, for checking the path
        :param module_name: The
        :param path:
        :param additional_args:
        :param num_cpus:
        :param num_gpus:
        :param memory_requirements:
        :param expected_duration:
        :return:
        """
        task = task_manager.get_import_dataset_task(
            module_name=module_name,
            path=path,
            additional_args=additional_args if additional_args is not None else {},
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_requirements=memory_requirements,
            expected_duration=expected_duration
        )
        if task.is_finished:
            self._datasets[name] = task.result
            self._set_property('datasets.{0}'.format(name), task.result)
        elif path_manager.check_path(path):
            task_manager.do_task(task)

    def import_benchmark(self, name: str, benchmark: arvet.core.benchmark.Benchmark,
                         db_client: arvet.database.client.DatabaseClient) -> None:
        """
        Import a benchmark, it will be used for all trials
        :param name: The name of the benchmark
        :param benchmark:
        :param db_client:
        :return:
        """
        if name not in self._benchmarks:
            self._benchmarks[name] = dh.add_unique(db_client.benchmarks_collection, benchmark)
            self._set_property('benchmarks.{0}'.format(name), self._benchmark_rpe)

    def serialize(self) -> dict:
        serialized = super().serialize()
        dh.add_schema_version(serialized, 'arvet:batch_analysis:simple_experiment:SimpleExperiment', 1)

        # Systems
        serialized['systems'] = self._systems

        # Image Sources
        serialized['datasets'] = self._datasets

        # Benchmarks
        serialized['benchmarks'] = self._benchmarks

        return serialized

    @classmethod
    def deserialize(cls, serialized_representation: dict,
                    db_client: arvet.database.client.DatabaseClient, **kwargs) -> 'SimpleExperiment':
        update_schema(serialized_representation, db_client)

        # Systems
        if 'systems' in serialized_representation:
            kwargs['systems'] = serialized_representation['systems']

        # Datasets
        if 'datasets' in serialized_representation:
            kwargs['datasets'] = serialized_representation['datasets']

        # Benchmarks
        if 'benchmarks' in serialized_representation:
            kwargs['benchmarks'] = serialized_representation['benchmarks']

        return super().deserialize(serialized_representation, db_client, **kwargs)


def update_schema(serialized: dict, db_client: arvet.database.client.DatabaseClient):
    # Clean out invalid ids
    if 'systems' in serialized:
        keys = list(serialized['systems'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.image_source_collection, serialized['systems'][key]):
                del serialized['systems'][key]

    if 'datasets' in serialized:
        keys = list(serialized['datasets'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.image_source_collection, serialized['datasets'][key]):
                del serialized['datasets'][key]

    if 'benchmarks' in serialized:
        keys = list(serialized['benchmarks'].keys())
        for key in keys:
            if not dh.check_reference_is_valid(db_client.image_source_collection, serialized['benchmarks'][key]):
                del serialized['benchmarks'][key]
