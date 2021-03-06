# Copyright (c) 2017, John Skinner
import abc
import typing
import logging
import bson
from pandas import DataFrame, read_pickle as pd_read_pickle
import pymodm
import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from pathlib import Path
import arvet.database.pymodm_abc as pymodm_abc
from arvet.database.autoload_modules import autoload_modules, get_model_classes
from arvet.database.reference_list_field import ReferenceListField
from arvet.core.system import VisionSystem, StochasticBehaviour
from arvet.core.image_source import ImageSource
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
from arvet.core.plot import Plot
import arvet.batch_analysis.task_manager as task_manager


class Experiment(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    A model for an experiment. The role of the experiment is to decide which systems should be run with
    which datasets, and which benchmarks should be used to measure them.
    They form a collection of results for us to analyse and write papers from

    Fundamentally, an experiment is a bunch of groups of ids for core object types,
    which will be mixed and matched in trials and benchmarks to produce our desired results.
    The groupings of the ids are meaningful for the particular experiment, to convey some level
    of association or meta-data that we can't articulate or encapsulate in the image_metadata.
    """
    name = fields.CharField(primary_key=True)
    enabled = fields.BooleanField(required=True, default=True)
    metric_results = ReferenceListField(MetricResult, blank=True, on_delete=fields.ReferenceField.PULL)
    plots = ReferenceListField(Plot, blank=True, on_delete=fields.ReferenceField.PULL)

    @abc.abstractmethod
    def schedule_tasks(self):
        """
        This is where you schedule the core tasks to train a system, run a system, or perform a benchmark,
        :return: void
        """
        pass

    def load_referenced_models(self):
        """
        Go through the models referenced by this experiment and ensure their model types have been loaded.
        This is necessary or accessing any of the reference fields will cause an exception.
        Clean references will access the metric results and plots, so we want to ensure that those models are loaded.
        :return:
        """
        # Load metric result models
        with no_auto_dereference(type(self)):
            model_ids = set(result_id for result_id in self.metric_results if isinstance(result_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(MetricResult, ids=list(model_ids))

        # Load plot models
        with no_auto_dereference(type(self)):
            model_ids = set(plot_id for plot_id in self.plots if isinstance(plot_id, bson.ObjectId))
        if len(model_ids) > 0:
            autoload_modules(Plot, ids=list(model_ids))

    def clean_references(self):
        """
        Clean up our lists of systems, image sources, and metrics.
        This fixes cases where a system, image source, or metric has been pulled from the list.
        The experiment may fail to save with nulls in its reference lists.
        :return:
        """
        self.metric_results = [metric_result for metric_result in self.metric_results if metric_result is not None]
        self.plots = [plot for plot in self.plots if plot is not None]

    def get_plots(self) -> typing.Set[str]:
        """
        Get a list of valid plots for this experiment.
        Each experiment pro
        :return:
        """
        return {
            plot.name
            for plot in self.plots
        }

    def plot_results(self, output_dir: Path, plot_names: typing.Collection[str] = None, display: bool = False,
                     cache_folder: Path = None):
        """
        Visualise the results from this experiment in different ways.

        :param output_dir: The base path to save the plots to.
        :param plot_names: The names of the plot to create, should only care about the values from get_plots
        :param display: Should the plot be displayed to the screen. Default false.
        :param cache_folder: A folder where the plot data is cached. Data will be loaded from there if available.
        :return: Nothing
        """
        # First, autoload the plot and result modules, to prevent a crash
        self.load_referenced_models()

        # Clean out null plots and other pulled references
        self.clean_references()

        # Filter the linked plots down to those selected
        if plot_names is None:
            plots = self.plots
        else:
            plots = [plot for plot in self.plots if plot.name in plot_names]
            extra_names = set(plot_names) - {plot.name for plot in self.plots}
            if len(extra_names) > 0:
                logging.getLogger(__name__).warning(f"Got unexpected plot names {extra_names}")

        if len(plots) > 0:
            # First, work out which properties our plots need
            columns = set(
                column
                for plot in plots
                for column in plot.get_required_columns()
            )
            if len(columns) <= 0:
                # No columns means no data and no plots
                logging.getLogger(__name__).info(f"No plots specified for {self.name}")
                return

            # The read the data from the results.
            data = None
            if cache_folder is not None:
                cache_file = self.get_cache_file(cache_folder, self.name)
                if cache_file.exists():
                    logging.getLogger(__name__).info(f"Reading plot data from cache file {cache_file}")
                    data = pd_read_pickle(cache_file)
                else:
                    logging.getLogger(__name__).warning(f"Cannot find cache file {cache_file}")
            if data is None:
                # Cache didn't exist, load the data from the database
                logging.getLogger(__name__).info("Reading plot data from database...")
                data = self.get_data(columns)

            # Make the output folder
            output_dir = (output_dir / self.name).resolve()
            output_dir.mkdir(exist_ok=True, parents=True)
            logging.getLogger(__name__).info(f"Outputting plots to {output_dir}")

            # Finally, delegate to the plots themselves to do the actual plotting
            for plot in plots:
                logging.getLogger(__name__).debug(f"Plotting {plot.name}...")
                try:
                    plot.plot_results(data, output_dir, display=display)
                except ValueError:
                    logging.getLogger(__name__).exception(f"Exception plotting {plot.name}")

    def cache_plot_data(self, cache_folder: Path):
        """
        Cache all the data necessary for producing all the plots in this experiment to file
        :param cache_folder: The folder to store the cache file
        :return:
        """
        # First, autoload the plot and result modules, to prevent a crash
        self.load_referenced_models()

        # Clean out null plots and other pulled references
        self.clean_references()

        # First, work out which properties our plots need. Use all the plots.
        columns = set(
            column
            for plot in self.plots
            for column in plot.get_required_columns()
        )
        if len(columns) <= 0:
            # No columns means no data and no plots
            logging.getLogger(__name__).info(f"No data to cache for {self.name}")
            return

        # Load the data to save
        logging.getLogger(__name__).info(f"Accumulating columns [{columns}] over "
                                         f"{len(self.metric_results)} results...")
        data = self.get_data(columns)
        logging.getLogger(__name__).info(f"Collected {len(data)} rows.")

        # Save the data to file
        cache_file = self.get_cache_file(cache_folder, self.name)
        if not cache_file.parent.exists():
            cache_file.parent.mkdir(parents=True)
        logging.getLogger(__name__).info(f"Saving data to {cache_file}")
        data.to_pickle(str(cache_file))

    def get_data(self, columns: typing.Iterable[str]) -> DataFrame:
        """
        Read the given columns into a dataframe.
        Collates data from all the metric results, based on a set of columns
        :param columns: The columns to get data for
        :return:        """
        data = []
        # Be a bit smarter about which results we load when, rather than accidentally loading and then keeping them all
        # We're still going to use so much memory, but at least it's only the fields we care about.
        with no_auto_dereference(type(self)):
            for result in self.metric_results:
                if isinstance(result, bson.ObjectId):
                    # Need to manually load the result, but we can get rid of it when we're done
                    result_obj = MetricResult.objects.get({'_id': result})
                    data += result_obj.get_results(columns)
                    del result_obj
                else:
                    data += result.get_results(columns)
        return DataFrame(data)

    @staticmethod
    def get_cache_file(cache_folder: Path, experiment_name: str) -> Path:
        """
        Choose the name of the file within a cache folder to store
        data underlying the plots.
        cache_plot_data will save to here, and plot_results will draw from here if available
        :param cache_folder: The folder in which the data will be cached.
        :param experiment_name: The name of the experiment to get the file for
        :return: A file within the cache folder where plot data will be stored
        """
        return cache_folder / (experiment_name.lower().replace(' ', '_') + '.pkl')

    @staticmethod
    def export_plot_data(experiment_name: str, cache_folder: Path) -> None:
        """
        Export the plot data for a particular experiment to the given folder
        :param experiment_name: The unique name of the experiment
        :param cache_folder: The folder to store the data in
        :return:
        """
        # Autoload the experiment type
        autoload_modules(Experiment, ids=[experiment_name])

        # Get the result ids from the experiment, bypassing the experiment object
        object_partials = list(
            Experiment.objects.raw({'_id': experiment_name}).only('metric_results', 'plots').values())
        metric_result_ids = [
            result_id
            for experiment_doc in object_partials
            for result_id in experiment_doc.get('metric_results', [])
        ]
        plot_ids = [
            plot_id
            for experiment_doc in object_partials
            for plot_id in experiment_doc.get('plots', [])
        ]
        if len(metric_result_ids) <= 0:
            logging.getLogger(__name__).info(f"No results for {experiment_name}, maybe re-run update_experiments?")
            return
        logging.getLogger(__name__).info(f"Caching plot data for {experiment_name} in {cache_folder} ...")

        # Get the columns
        columns = get_columns_for_plots(plot_ids)
        if len(columns) <= 0:
            # No columns means no data and no plots
            logging.getLogger(__name__).info(f"No data to cache for {experiment_name}, are there any plots?")
            return

        # Load the metric result types
        autoload_modules(MetricResult, ids=metric_result_ids)

        # Read all the results
        logging.getLogger(__name__).info(f"Accumulating columns [{columns}] over {len(metric_result_ids)} results...")
        data = []
        for result_id in metric_result_ids:
            try:
                metric_result = MetricResult.objects.get({'_id': result_id})
            except MetricResult.DoesNotExist:
                continue
            if metric_result is not None:
                data.extend(metric_result.get_results(columns))
            del metric_result
        data = DataFrame(data)
        logging.getLogger(__name__).info(f"Collected {len(data)} rows.")

        # Save the data to file
        cache_file = Experiment.get_cache_file(cache_folder, experiment_name)
        if not cache_file.parent.exists():
            cache_file.parent.mkdir(parents=True)
        logging.getLogger(__name__).info(f"Saving data to {cache_file}")
        data.to_pickle(str(cache_file))


def run_all(
        systems: typing.Iterable[typing.Union[VisionSystem, bson.ObjectId]],
        image_sources: typing.Iterable[typing.Union[ImageSource, bson.ObjectId]],
        repeats: int,
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> typing.Tuple[typing.Mapping[bson.ObjectId, typing.Mapping[bson.ObjectId, typing.List[bson.ObjectId]]], int]:
    """
    Run all the systems on all the image sources.
    Returns the trial results, grouped by system and image source
    :param systems: The set of systems to run
    :param image_sources: The set of image sources to run it on
    :param repeats:
    :param num_cpus:
    :param num_gpus:
    :param memory_requirements:
    :param expected_duration:
    :return: A map of system id and image source id to trial results, and the number of new trial results still to come
    """
    remaining = 0
    trial_results = {}
    for system in systems:
        # If the system is just an id, load the object
        if isinstance(system, bson.ObjectId):
            system = VisionSystem.objects.get({'_id': system})
        for image_source in image_sources:
            # If the image source is just an ID, load the object, minimally
            if isinstance(image_source, bson.ObjectId):
                try:
                    image_source = load_minimal_image_source(image_source)
                except RuntimeError:
                    logging.getLogger(__name__).exception(f"Failed to load image source {image_source}, skipping")
                    continue

            if system.is_image_source_appropriate(image_source):
                # If the system is deterministic, run it once
                actual_repeats = repeats if system.is_deterministic() is not StochasticBehaviour.DETERMINISTIC else 1
                for repeat in range(actual_repeats):
                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        num_cpus=num_cpus, num_gpus=num_gpus,
                        memory_requirements=memory_requirements,
                        expected_duration=expected_duration
                    )
                    if task.is_finished:
                        result = task.result_id
                        if result is None:
                            # Task was finished, but has since been deleted. Re-create the task as unfinished.
                            task.delete()
                            task = task_manager.get_run_system_task(
                                system=system,
                                image_source=image_source,
                                repeat=repeat,
                                num_cpus=num_cpus, num_gpus=num_gpus,
                                memory_requirements=memory_requirements,
                                expected_duration=expected_duration
                            )
                            remaining += 1
                            task.save()
                        else:
                            # Got a valid trial result
                            if system.identifier not in trial_results:
                                trial_results[system.identifier] = {}
                            if image_source.identifier not in trial_results[system.identifier]:
                                trial_results[system.identifier][image_source.identifier] = []
                            trial_results[system.identifier][image_source.identifier].append(result)
                    else:
                        remaining += 1
                        changed = False
                        if task.num_cpus != num_cpus:
                            task.num_cpus = num_cpus
                            changed = True
                        if task.num_gpus != num_gpus:
                            task.num_gpus = num_gpus
                            changed = True
                        if task.memory_requirements != memory_requirements:
                            task.memory_requirements = memory_requirements
                            changed = True
                        if task.expected_duration != expected_duration:
                            task.expected_duration = expected_duration
                            changed = True
                        if changed or task.pk is None:
                            task.save()
    return trial_results, remaining


def run_all_with_seeds(
        systems: typing.Iterable[typing.Union[VisionSystem, bson.ObjectId]],
        image_sources: typing.Iterable[typing.Union[ImageSource, bson.ObjectId]],
        seeds: typing.Iterable[int],
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> typing.Tuple[typing.Mapping[bson.ObjectId, typing.Mapping[bson.ObjectId, typing.List[bson.ObjectId]]], int]:
    """
    Run all the systems on all the image sources.
    Returns the trial results, grouped by system and image source
    :param systems: The set of systems to run
    :param image_sources: The set of image sources to run it on
    :param seeds:
    :param num_cpus:
    :param num_gpus:
    :param memory_requirements:
    :param expected_duration:
    :return: A map of system id and image source id to trial results, and the number of new trial results still to come
    """
    remaining = 0
    trial_results = {}
    for system in systems:
        # If the system is just an id, load the object
        if isinstance(system, bson.ObjectId):
            system = VisionSystem.objects.get({'_id': system})
        for image_source in image_sources:
            # If the image source is just an ID, load the object, minimally
            if isinstance(image_source, bson.ObjectId):
                try:
                    image_source = load_minimal_image_source(image_source)
                except RuntimeError:
                    logging.getLogger(__name__).exception(f"Failed to load image source {image_source}, skipping")
                    continue

            if system.is_image_source_appropriate(image_source):
                for seed in seeds:
                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=0,
                        seed=seed,
                        num_cpus=num_cpus, num_gpus=num_gpus,
                        memory_requirements=memory_requirements,
                        expected_duration=expected_duration
                    )
                    if task.is_finished:
                        result = task.result_id
                        if result is None:
                            # Task finished, but the result has since been deleted. Delete and re-create the task
                            task.delete()
                            task = task_manager.get_run_system_task(
                                system=system,
                                image_source=image_source,
                                repeat=0,
                                seed=seed,
                                num_cpus=num_cpus, num_gpus=num_gpus,
                                memory_requirements=memory_requirements,
                                expected_duration=expected_duration
                            )
                            remaining += 1
                            task.save()
                        else:
                            # Job is done, add the trial result to the map
                            if system.pk not in trial_results:
                                trial_results[system.pk] = {}
                            if image_source.pk not in trial_results[system.pk]:
                                trial_results[system.pk][image_source.pk] = []
                            trial_results[system.pk][image_source.pk].append(result)
                    else:
                        remaining += 1
                        changed = False
                        if task.num_cpus != num_cpus:
                            task.num_cpus = num_cpus
                            changed = True
                        if task.num_gpus != num_gpus:
                            task.num_gpus = num_gpus
                            changed = True
                        if task.memory_requirements != memory_requirements:
                            task.memory_requirements = memory_requirements
                            changed = True
                        if task.expected_duration != expected_duration:
                            task.expected_duration = expected_duration
                            changed = True
                        if changed or task.pk is None:
                            task.save()
    return trial_results, remaining


def measure_all(
        metrics: typing.Iterable[typing.Union[Metric, bson.ObjectId]],
        trial_result_groups: typing.Iterable[typing.List[typing.Union[TrialResult, bson.ObjectId]]],
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> typing.Tuple[typing.Mapping[bson.ObjectId, typing.List[bson.ObjectId]], int]:
    """

    :param metrics:
    :param trial_result_groups:
    :param num_cpus:
    :param num_gpus:
    :param memory_requirements:
    :param expected_duration:
    :return:
    """
    remaining = 0
    metric_results = {}
    for metric in metrics:
        # If the metric is just an id, load the object
        if isinstance(metric, bson.ObjectId):
            metric = Metric.objects.get({'_id': metric})
        for trial_results in trial_result_groups:
            # Load minimal trial result objects for the missing trial results
            trial_results = [
                load_minimal_trial_result(trial_result) if isinstance(trial_result, bson.ObjectId) else trial_result
                for trial_result in trial_results
            ]
            if all(metric.is_trial_appropriate(trial_result) for trial_result in trial_results):
                task = task_manager.get_measure_trial_task(
                    trial_results=trial_results,
                    metric=metric,
                    num_cpus=num_cpus, num_gpus=num_gpus,
                    memory_requirements=memory_requirements,
                    expected_duration=expected_duration
                )
                if task.is_finished:
                    result = task.result_id
                    if result is None:
                        # Metric result was since deleted, delete and re-create the task
                        task.delete()
                        task = task_manager.get_measure_trial_task(
                            trial_results=trial_results,
                            metric=metric,
                            num_cpus=num_cpus, num_gpus=num_gpus,
                            memory_requirements=memory_requirements,
                            expected_duration=expected_duration
                        )
                        task.save()
                        remaining += 1
                    else:
                        if metric.identifier not in metric_results:
                            metric_results[metric.identifier] = []
                        metric_results[metric.identifier].append(result)
                else:
                    remaining += 1
                    changed = False
                    if task.num_cpus != num_cpus:
                        task.num_cpus = num_cpus
                        changed = True
                    if task.num_gpus != num_gpus:
                        task.num_gpus = num_gpus
                        changed = True
                    if task.memory_requirements != memory_requirements:
                        task.memory_requirements = memory_requirements
                        changed = True
                    if task.expected_duration != expected_duration:
                        task.expected_duration = expected_duration
                        changed = True
                    if changed or task.pk is None:
                        task.save()
    return metric_results, remaining


def load_minimal_image_source(image_source_id: bson.ObjectId) -> ImageSource:
    """
    Load a minimally sized image source object from its ID
    :param image_source_id:
    :return:
    """
    image_source_type = get_model_classes(ImageSource, [image_source_id])
    if len(image_source_type) != 1:
        raise RuntimeError("Got multiple image source types for id \"{0}\", got types {1}".format(
            image_source_id, image_source_type))
    image_source_type = image_source_type[0]
    return image_source_type.load_minimal(image_source_id)


def load_minimal_trial_result(image_source_id: bson.ObjectId) -> TrialResult:
    """
    Load a minimally sized image source object from its ID
    :param image_source_id:
    :return:
    """
    trial_type = get_model_classes(TrialResult, [image_source_id])
    if len(trial_type) != 1:
        raise RuntimeError("Got multiple trial result types for id \"{0}\", got types {1}".format(
            image_source_id, trial_type))
    trial_type = trial_type[0]
    return trial_type.load_minimal(image_source_id)


def get_columns_for_plots(plot_ids: typing.Iterable[bson.ObjectId]) -> typing.Set[str]:
    """
    For a set of plot ids, get the required columns for those plots.
    Handles auto-loading the plot types, and the plot objects themselves will go out of scope once this helper completes
    :param plot_ids: The list of plot ids to search for
    :return: A combined set of the required columns for all the given plots
    """
    plot_ids = list(plot_ids)
    autoload_modules(Plot, ids=plot_ids)
    plots = Plot.objects.raw({'_id': {'$in': plot_ids}})
    columns = set(
        column
        for plot in plots
        for column in plot.get_required_columns()
    )
    return columns
