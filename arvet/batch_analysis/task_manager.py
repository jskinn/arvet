# Copyright (c) 2017, John Skinner
import bson
import typing

from arvet.database.autoload_modules import get_model_classes
from arvet.core.image_source import ImageSource
from arvet.core.system import VisionSystem, StochasticBehaviour
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric
from arvet.core.trial_comparison import TrialComparisonMetric

from arvet.batch_analysis.task import Task, JobState
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask
from arvet.batch_analysis.tasks.compare_trials_task import CompareTrialTask
from arvet.batch_analysis.job_system import JobSystem


def get_import_dataset_task(
        module_name: str, path: str, additional_args: dict = None,
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> ImportDatasetTask:
    """
    Get a task to import a dataset.
    Most of the parameters are resources requirements passed to the job system.
    :param module_name: The name of the python module to use to do the import as a string.
    It must have a function 'import_dataset', taking a directory and the database client
    :param path: The root file or directory describing the dataset to import
    :param additional_args: Additional arguments to the importer module, depends on what module is chosen
    :param num_cpus: The number of CPUs required for the job. Default 1.
    :param num_gpus: The number of GPUs required for the job. Default 0.
    :param memory_requirements: The memory required for this job. Default 3 GB.
    :param expected_duration: The expected time this job will take. Default 1 hour.
    :return: An ImportDatasetTask containing the task state.
    """
    if additional_args is None:
        additional_args = {}

    try:
        return ImportDatasetTask.objects.get({
            'module_name': str(module_name),
            'path': str(path),
            'additional_args': additional_args
        })
    except ImportDatasetTask.DoesNotExist:
        return ImportDatasetTask(
            module_name=str(module_name),
            path=str(path),
            additional_args=additional_args,
            state=JobState.UNSTARTED,
            num_cpus=int(num_cpus),
            num_gpus=int(num_gpus),
            memory_requirements=str(memory_requirements),
            expected_duration=str(expected_duration)
        )


def get_run_system_task(
        system: typing.Union[VisionSystem, bson.ObjectId],
        image_source: typing.Union[ImageSource, bson.ObjectId],
        repeat: int = 0, seed: int = 0,
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> RunSystemTask:
    """
    Get a task to run a system.
    Most of the parameters are resources requirements passed to the job system.
    :param system: The id of the vision system to test
    :param image_source: The id of the image source to test with
    :param repeat: The repeat of this trial, so we can run the same system more than once.
    :param seed: The random seed to use. Ignored for anything except SEEDED systems.
    :param num_cpus: The number of CPUs required for the job. Default 1.
    :param num_gpus: The number of GPUs required for the job. Default 0.
    :param memory_requirements: The memory required for this job. Default 3 GB.
    :param expected_duration: The expected time this job will take. Default 1 hour.
    :return: A RunSystemTask
    """
    if isinstance(system, VisionSystem):
        use_seed = (system.is_deterministic() is StochasticBehaviour.SEEDED)
        system = system.identifier
    else:
        if VisionSystem.objects.raw({'_id': system}).count() < 1:
            raise ValueError("system is not a valid VisionSystem id")
        system_classes = get_model_classes(VisionSystem, [system])
        if len(system_classes) >= 1:
            use_seed = (system_classes[0].is_deterministic() is StochasticBehaviour.SEEDED)
        else:
            raise ValueError("Could not load class for system {0}".format(system))
    if isinstance(image_source, ImageSource):
        image_source = image_source.identifier
    elif ImageSource.objects.raw({'_id': image_source}).count() < 1:
        raise ValueError("image_source is not a valid ImageSource id")

    query = {
        'system': system,
        'image_source': image_source,
        'repeat': int(repeat)
    }
    if use_seed:
        query['seed'] = int(seed)

    try:
        return RunSystemTask.objects.get(query)
    except RunSystemTask.DoesNotExist:
        return RunSystemTask(
            system=system,
            image_source=image_source,
            repeat=int(repeat),
            seed=int(seed) if use_seed else None,
            state=JobState.UNSTARTED,
            num_cpus=int(num_cpus),
            num_gpus=int(num_gpus),
            memory_requirements=str(memory_requirements),
            expected_duration=str(expected_duration)
        )


def get_measure_trial_task(
        trial_results: typing.Union[typing.Iterable[TrialResult], typing.Iterable[bson.ObjectId]],
        metric: typing.Union[Metric, bson.ObjectId],
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> MeasureTrialTask:
    """
    Get a task to benchmark a trial result.
    Most of the parameters are resources requirements passed to the job system.
    :param trial_results: The ids of the trial results to benchmark
    :param metric: The id of the benchmark to use
    :param num_cpus: The number of CPUs required for the job. Default 1.
    :param num_gpus: The number of GPUs required for the job. Default 0.
    :param memory_requirements: The memory required for this job. Default 3 GB.
    :param expected_duration: The expected time this job will take. Default 1 hour.
    :return: A BenchmarkTrialTask
    """
    trial_results = [
        trial_result.identifier if isinstance(trial_result, TrialResult) else trial_result
        for trial_result in trial_results
    ]
    if TrialResult.objects.raw({'_id': {'$in': trial_results}}).count() < len(trial_results):
        raise ValueError('trial_results contains invalid trial result id')
    if isinstance(metric, Metric):
        metric = metric.identifier
    elif Metric.objects.raw({'_id': metric}).count() < 1:
        raise ValueError("metric is not a valid Metric id")
    try:
        return MeasureTrialTask.objects.get({
            'trial_results': {'$all': trial_results},
            'metric': metric
        })
    except MeasureTrialTask.DoesNotExist:
        return MeasureTrialTask(
            trial_results=trial_results,
            metric=metric,
            state=JobState.UNSTARTED,
            num_cpus=int(num_cpus),
            num_gpus=int(num_gpus),
            memory_requirements=str(memory_requirements),
            expected_duration=str(expected_duration)
        )


def get_trial_comparison_task(
        trial_results_1: typing.Union[typing.Iterable[TrialResult], typing.Iterable[bson.ObjectId]],
        trial_results_2: typing.Union[typing.Iterable[TrialResult], typing.Iterable[bson.ObjectId]],
        comparison_metric: typing.Union[TrialComparisonMetric, bson.ObjectId],
        num_cpus: int = 1, num_gpus: int = 0,
        memory_requirements: str = '3GB', expected_duration: str = '1:00:00'
) -> CompareTrialTask:
    """
    Get a task to compare two trial results.
    Most of the parameters are resources requirements passed to the job system.
    :param trial_results_1: The ids of the trial results to compare
    :param trial_results_2: The id of the reference trial results to compare
    :param comparison_metric: The id of the comparison benchmark to use
    :param num_cpus: The number of CPUs required for the job. Default 1.
    :param num_gpus: The number of GPUs required for the job. Default 0.
    :param memory_requirements: The memory required for this job. Default 3 GB.
    :param expected_duration: The expected time this job will take. Default 1 hour.
    :return: A CompareTrialTask
    """
    trial_results_1 = [
        trial_result.identifier if isinstance(trial_result, TrialResult) else trial_result
        for trial_result in trial_results_1
    ]
    trial_results_2 = [
        trial_result.identifier if isinstance(trial_result, TrialResult) else trial_result
        for trial_result in trial_results_2
    ]
    if TrialResult.objects.raw({'_id': {'$in': trial_results_1}}).count() < len(trial_results_1):
        raise ValueError('trial_results_1 contains invalid trial result id')
    if TrialResult.objects.raw({'_id': {'$in': trial_results_2}}).count() < len(trial_results_2):
        raise ValueError('trial_results_2 contains invalid trial result id')
    if isinstance(comparison_metric, TrialComparisonMetric):
        comparison_metric = comparison_metric.identifier
    elif TrialComparisonMetric.objects.raw({'_id': comparison_metric}).count() < 1:
        raise ValueError("metric is not a valid Metric id")

    try:
        return CompareTrialTask.objects.get({
            'trial_results_1': trial_results_1,
            'trial_results_2': trial_results_2,
            'metric': comparison_metric
        })
    except CompareTrialTask.DoesNotExist:
        return CompareTrialTask(
            trial_results_1=trial_results_1,
            trial_results_2=trial_results_2,
            metric=comparison_metric,
            state=JobState.UNSTARTED,
            num_cpus=int(num_cpus),
            num_gpus=int(num_gpus),
            memory_requirements=str(memory_requirements),
            expected_duration=str(expected_duration)
        )


def schedule_tasks(job_system: JobSystem,
                   task_ids: typing.Iterable[bson.ObjectId] = None):
    """
    Schedule all pending tasks using the provided job system
    This should both star
    :param job_system: The job system used to run the tasks
    :param task_ids: A limited set of job ids to schedule. If None, schedule all possible.
    :return:
    """
    # First, check the jobs that should already be running on this node
    all_running = Task.objects.raw({
        'state': JobState.RUNNING.name,
        'node_id': job_system.node_id
    })
    for task in all_running:
        if not job_system.is_job_running(task.job_id):
            # Task should be running, but job system says it isn't re-run
            task.mark_job_failed()
            task.save()

    # Then, schedule all the unscheduled tasks
    query = {'state': JobState.UNSTARTED.name}
    if task_ids is not None:
        query['_id'] = {'$in': list(task_ids)}
    all_available = Task.objects.raw(query)
    for task in all_available:
        job_system.run_task(task)


def count_pending_tasks() -> int:
    """
    Get the number of task objects that are not marked as done.
    :return:
    """
    return Task.objects.raw({'state': {'$ne': JobState.DONE.name}}).count()
