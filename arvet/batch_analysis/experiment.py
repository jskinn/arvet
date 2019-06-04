# Copyright (c) 2017, John Skinner
import abc
import os.path
import typing
import bson
import pymodm
import pymodm.fields as fields
import arvet.database.pymodm_abc as pymodm_abc
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
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

    enabled = fields.BooleanField(required=True)
    metric_results = fields.ListField(
        fields.ReferenceField(MetricResult, on_delete=fields.ReferenceField.PULL)
    )

    @abc.abstractmethod
    def schedule_tasks(self):
        """
        This is where you schedule the core tasks to train a system, run a system, or perform a benchmark,
        :return: void
        """
        pass

    def plot_results(self):
        """
        Visualise the results from this experiment.
        Non-compulsory, but will be called from plot_results.py
        :return:
        """
        pass

    def export_data(self):
        """
        Allow experiments to export some data, usually to file.
        I'm currently using this to dump camera trajectories so I can build simulations around them,
        but there will be other circumstances where we want to. Naturally, this is optional.
        :return:
        """
        pass

    def perform_analysis(self):
        """
        Method by which an experiment can perform large-scale analysis as a job.
        This is what is called by AnalyseResultsTask
        This should save it's output somehow, preferably within the output folder given by get_output_folder()
        :return:
        """
        pass

    @classmethod
    def get_output_folder(cls):
        """
        Get a unique output folder for this experiment.
        Really, we just name the folder after the experiment, but it's nice to do this in a standardized way.
        :return:
        """
        return os.path.join('results', cls.__name__)


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
                        trial_results[system.identifier][image_source.identifier].append(task.result)
                    else:
                        remaining += 1
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
                    metric_results[metric.identifier].append(task.result)
                else:
                    remaining += 1
                    task.save()
    return metric_results, remaining
