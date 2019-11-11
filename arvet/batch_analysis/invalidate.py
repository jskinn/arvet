import typing
import logging
import bson
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
from arvet.core.image_source import ImageSource
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask


def invalidate_dataset_loaders(loader_modules: typing.Iterable[str]):
    """
    Invalidate all image collections created by importing using a particular dataset loader module.
    Useful when there are bugs in the dataset loader
    :param loader_modules:
    :return:
    """
    loader_module_list = list(loader_modules)
    # Step 1: Find all tasks with this module and invalidate the collections
    with no_auto_dereference(ImportDatasetTask):
        image_collection_ids = [task.result for task in ImportDatasetTask.objects.raw({
            'module_name': {'$in': loader_module_list}, 'result': {'$exists': True}
        }).only('result')]

    # Step 2: Delete all the associated image collections
    invalidate_image_collections(image_collection_ids)

    # Step 2: Remove all the tasks with this dataset loader
    ImportDatasetTask.objects.raw({
        'module_name': {'$in': loader_module_list}
    }).delete()


def invalidate_image_collections(image_source_ids: typing.Iterable[bson.ObjectId]) -> int:
    """
    Invalidate the data associated with a particular image source.
    This cascades to derived trial results, and from there to benchmark results.
    Also cleans out tasks.
    :param image_source_ids: The ids of the image sources to remove
    :return:
    """
    # Just delete the image collections, reference fields should handle the rest
    image_source_ids = list(image_source_ids)
    autoload_modules(ImageSource, image_source_ids)
    removed = ImageSource.objects.raw({
        '_id': {'$in': image_source_ids}
    }).delete()
    logging.getLogger(__name__).info("removed {0} image sources".format(removed))
    return removed


def invalidate_systems(system_ids: typing.Iterable[bson.ObjectId]) -> int:
    system_ids = list(system_ids)
    autoload_modules(VisionSystem, system_ids)
    removed = VisionSystem.objects.raw({
        '_id': {'$in': system_ids}
    }).delete()
    logging.getLogger(__name__).info("removed {0} systems".format(removed))
    return removed


def invalidate_systems_by_name(system_names: typing.Iterable[str]) -> int:
    system_names = list(system_names)
    autoload_modules(VisionSystem, classes=system_names)
    removed = VisionSystem.objects.raw({
        '_cls': {'$in': system_names}
    }).delete()
    logging.getLogger(__name__).info("removed {0} systems".format(removed))
    return removed


def invalidate_trial_results(trial_result_ids: typing.Iterable[bson.ObjectId]) -> int:
    trial_result_ids = list(trial_result_ids)
    autoload_modules(TrialResult, trial_result_ids)
    removed = TrialResult.objects.raw({
        '_id': {'$in': trial_result_ids}
    }).delete()
    logging.getLogger(__name__).info("removed {0} trial results".format(removed))
    return removed


def invalidate_failed_trial_results() -> int:
    autoload_modules(TrialResult)
    removed = TrialResult.objects.raw({
        'success': False
    }).delete()
    logging.getLogger(__name__).info("removed {0} trial results".format(removed))
    return removed


def invalidate_metrics(metric_ids: typing.Iterable[bson.ObjectId]) -> int:
    metric_ids = list(metric_ids)
    autoload_modules(Metric, metric_ids)
    removed = Metric.objects.raw({
        '_id': {'$in': metric_ids}
    }).delete()
    logging.getLogger(__name__).info("removed {0} metrics".format(removed))
    return removed


def invalidate_metric_results(metric_result_ids: typing.Iterable[bson.ObjectId]) -> int:
    metric_result_ids = list(metric_result_ids)
    autoload_modules(MetricResult, metric_result_ids)
    removed = MetricResult.objects.raw({
        '_id': {'$in': metric_result_ids}
    }).delete()
    logging.getLogger(__name__).info("removed {0} metric results".format(removed))
    return removed


def invalidate_failed_metric_results() -> int:
    autoload_modules(MetricResult)
    removed = MetricResult.objects.raw({
        'success': False
    }).delete()
    logging.getLogger(__name__).info("removed {0} metric results".format(removed))
    return removed
