# Copyright (c) 2019, John Skinner
import typing
import bson
from pymodm.context_managers import no_auto_dereference
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
from arvet.core.trial_comparison import TrialComparisonMetric, TrialComparisonResult

from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
import arvet.batch_analysis.task_manager as task_manager


def schedule_all(
        systems: typing.Iterable[VisionSystem],
        image_sources: typing.Iterable[ImageSource],
        metrics: typing.Iterable[Metric],
        repeats: int = 1,
        allow_metrics_on_incomplete: bool = False
) -> int:
    """
    Schedule all combinations of running some group of systems with some group of image sources,
    and then measuring the results with some list of metrics.
    Uses is_image_source_appropriate and is_benchmark_appropriate to filter.
    Created results are added to the metric_results property

    :param systems: The list of systems to test
    :param image_sources: The list of image sources to use
    :param metrics: The list of benchmark to measure the results
    :param repeats: The number of times to repeat
    :param allow_metrics_on_incomplete: Whether to run benchmarks when not all the trials have completed yet.
    :return: the number of anticipated changes when all current tasks are done
    """
    # Trial results will be collected as we go
    trial_result_groups = []
    repeats = max(repeats, 1)  # always at least 1 repeat
    anticipated_changes = 0

    # For each image dataset, run libviso with that dataset, and store the result in the trial map
    for image_source in image_sources:
        for system in systems:
            if system.is_image_source_appropriate(image_source):
                trial_result_group = set()
                for repeat in range(repeats):
                    task = task_manager.get_run_system_task(
                        system=system,
                        image_source=image_source,
                        repeat=repeat,
                        expected_duration='8:00:00',
                        memory_requirements='12GB'
                    )
                    if not task.is_finished:
                        anticipated_changes += 1
                        if task.identifier is None:
                            task.save()
                    else:
                        with no_auto_dereference(RunSystemTask):
                            trial_result_group.add(task.result)

                if len(trial_result_group) >= repeats or allow_metrics_on_incomplete:
                    # Schedule benchmarks for the group
                    trial_result_groups.append(trial_result_group)

    # measure trial results collected in the previous step
    for metric in metrics:
        for trial_result_group in trial_result_groups:
            if are_trials_appropriate(metric, trial_result_group):
                task = task_manager.get_measure_trial_task(
                    trial_results=trial_result_group,
                    metric=metric,
                    expected_duration='6:00:00',
                    memory_requirements='6GB'
                )
                if not task.is_finished:
                    anticipated_changes += 1
                    if task.identifier is None:
                        task.save()
    return anticipated_changes


def get_trial_results(system_id: bson.ObjectId, image_source_id: bson.ObjectId) \
        -> typing.Iterable[TrialResult]:
    """
    Get the trial result produced by running a given system with a given image source.
    Return None if the system has not been run with that image source.
    :param system_id: The id of the system
    :param image_source_id: The id of the image source
    :return: The trial results, or None if the trial has not been performed
    """
    return TrialResult.objects.raw({
        'system': system_id,
        'image_source': image_source_id
    }).all()


def get_metric_result(system_id: bson.ObjectId, image_source_id: bson.ObjectId,
                      metric_id: bson.ObjectId) -> typing.Union[MetricResult, None]:
    """
    Get the results of benchmarking the results of a particular system on a particular image source,
    using the given benchmark
    :param system_id: The id of the system used to produce the benchmarked trials
    :param image_source_id: The id of the image source to perform the benchmarked trials
    :param metric_id: The id of the benchmark used
    :return: The id of the result object, or None if the trials have not been measured.
    """
    trial_results = list(TrialResult.objects.raw({
        'system': system_id,
        'image_source': image_source_id
    }).only('_id'))
    return MetricResult.objects.get({
        'metric': metric_id,
        'trial_results': {'$all': trial_results}
    })


def are_trials_appropriate(metric: Metric, trial_result_ids: typing.Iterable[bson.ObjectId]) -> bool:
    """
    Check if a given set of trial results can be measured using a particular metric
    :param metric:
    :param trial_result_ids:
    :return:
    """
    for trial_result_id in trial_result_ids:
        try:
            trial_result = TrialResult.objects.get({'_id': trial_result_id})
        except TrialResult.DoesNotExist:
            return False
        if not metric.is_trial_appropriate(trial_result):
            return False
    return True
