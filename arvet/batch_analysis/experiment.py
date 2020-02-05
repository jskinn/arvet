# Copyright (c) 2017, John Skinner
import abc
import typing
import bson
from pandas import DataFrame
import pymodm
import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from pathlib import PurePath
import arvet.database.pymodm_abc as pymodm_abc
from arvet.database.autoload_modules import autoload_modules
from arvet.database.reference_list_field import ReferenceListField
from arvet.core.system import VisionSystem
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
    metric_results = ReferenceListField(MetricResult, on_delete=fields.ReferenceField.PULL)
    plots = ReferenceListField(Plot, on_delete=fields.ReferenceField.PULL)

    @abc.abstractmethod
    def schedule_tasks(self):
        """
        This is where you schedule the core tasks to train a system, run a system, or perform a benchmark,
        :return: void
        """
        pass

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

    def plot_results(self, output_dir: PurePath, plot_names: typing.Container[str] = None, display: bool = False):
        """
        Visualise the results from this experiment in different ways.

        :param output_dir: The base path to save the plots to.
        :param plot_names: The names of the plot to create, should only care about the values from get_plots
        :param display: Should the plot be displayed to the screen. Default false.
        :return: Nothing
        """
        # First, autoload the plot modules
        with no_auto_dereference(type(self)):
            plot_ids = set(plot_id for plot_id in self.plots if isinstance(plot_id, bson.ObjectId))
        if len(plot_ids) > 0:
            autoload_modules(Plot, ids=list(plot_ids))

        # Filter the linked plots down to those selected
        if plot_names is None:
            plots = self.plots
        else:
            plots = [plot for plot in self.plots if plot.name in plot_names]

        if len(plots) > 0:
            # First, work out which properties our plots need
            columns = set(
                column
                for plot in plots
                for column in plot.get_required_columns()
            )
            if len(columns) <= 0:
                # No columns means no data and no plots
                return

            # The read the data from the results.
            data = self.get_data(columns)

            # Make the output folder
            output_dir = (output_dir / self.name).resolve()
            output_dir.mkdir(exist_ok=True, parents=True)

            # Finally, delegate to the plots themselves to do the actual plotting
            for plot in plots:
                plot.plot_results(data, output_dir, display=display)

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


def run_all(
        systems: typing.Iterable[VisionSystem],
        image_sources: typing.Iterable[ImageSource],
        repeats: int,
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> typing.Tuple[typing.Mapping[bson.ObjectId, typing.Mapping[bson.ObjectId, typing.List[TrialResult]]], int]:
    """
    Run all the systems on all the image sources.
    Returns the trial results, grouped by system and image source
    :param systems:
    :param image_sources:
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
        for image_source in image_sources:
            if system.is_image_source_appropriate(image_source):
                for repeat in range(repeats):
                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        num_cpus=num_cpus, num_gpus=num_gpus,
                        memory_requirements=memory_requirements,
                        expected_duration=expected_duration
                    )
                    if task.is_finished:
                        if system.identifier not in trial_results:
                            trial_results[system.identifier] = {}
                        if image_source.identifier not in trial_results[system.identifier]:
                            trial_results[system.identifier][image_source.identifier] = []
                        trial_results[system.identifier][image_source.identifier].append(task.get_result())
                    else:
                        remaining += 1
                        if task.pk is None:
                            task.save()
    return trial_results, remaining


def measure_all(
        metrics: typing.Iterable[Metric],
        trial_result_groups: typing.Iterable[typing.List[TrialResult]],
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> typing.Tuple[typing.Mapping[bson.ObjectId, typing.List[MetricResult]], int]:
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
        for trial_results in trial_result_groups:
            if all(metric.is_trial_appropriate(trial_result) for trial_result in trial_results):
                task = task_manager.get_measure_trial_task(
                    trial_results=trial_results,
                    metric=metric,
                    num_cpus=num_cpus, num_gpus=num_gpus,
                    memory_requirements=memory_requirements,
                    expected_duration=expected_duration
                )
                if task.is_finished:
                    if metric.identifier not in metric_results:
                        metric_results[metric.identifier] = []
                    metric_results[metric.identifier].append(task.get_result())
                else:
                    remaining += 1
                    if task.pk is None:
                        task.save()
    return metric_results, remaining
