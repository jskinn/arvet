#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
import bson
import typing
import re

from arvet.config.global_configuration import load_global_config
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
import arvet.batch_analysis.invalidate as invalidate


def is_id(val: str) -> bool:
    """
    Is a given string a bson id. It should be a string of hex digits
    :param val:
    :return:
    """
    return re.fullmatch('[0-9a-f]+', val) is not None


def main(
        systems: typing.List[str] = None, datasets: typing.List[str] = None,
        image_collections: typing.List[str] = None,
        # simulators: typing.List[str] = None,
        # controllers: typing.List[str] = None,
        trial_results: typing.List[str] = None,
        metrics: typing.List[str] = None, metric_results: typing.List[str] = None,
        failed_trials: bool = False,
        failed_metrics: bool = False,
        incomplete_tasks: bool = False,
        mongodb_host: str = None,
        mongodb_port: int = None
):
    """
    Command line control to invalidate various objects.
    :args: Each of the different types of thing to invalidate
    :return:
    """
    if systems is None:
        systems = []
    if datasets is None:
        datasets = []
    if image_collections is None:
        image_collections = []
    # if simulators is None:
    #     simulators = []
    # if controllers is None:
    #     controllers = []
    if trial_results is None:
        trial_results = []
    if metrics is None:
        metrics = []
    if metric_results is None:
        metric_results = []

    # Sanitize arguments
    system_ids = [bson.ObjectId(oid) for oid in systems if is_id(oid)]
    system_names = [name for name in systems if not is_id(name)]
    image_collections = [bson.ObjectId(oid) for oid in image_collections if is_id(oid)]
    # controllers = [bson.ObjectId(oid) for oid in controllers if is_id(oid)]
    trial_results = [bson.ObjectId(oid) for oid in trial_results if is_id(oid)]
    metrics = [bson.ObjectId(oid) for oid in metrics if is_id(oid)]
    metric_results = [bson.ObjectId(oid) for oid in metric_results if is_id(oid)]

    # Load the configuration
    config = load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])

    # Configure the database and the image manager
    dbconn.configure(config['database'], override_host=mongodb_host, override_port=mongodb_port)
    im_manager.configure(config['image_manager'])

    # Invalidate systems
    if len(system_ids) > 0:
        logging.getLogger(__name__).info("Invalidating systems {0}".format(system_ids))
        invalidate.invalidate_systems(system_ids)
    if len(system_names) > 0:
        logging.getLogger(__name__).info("Invalidating systems {0}".format(system_names))
        invalidate.invalidate_systems_by_name(system_names)

    # Invalidate datasets by loader module
    if len(datasets) > 0:
        logging.getLogger(__name__).info("Invalidating dataset loaders {0}".format(datasets))
        invalidate.invalidate_dataset_loaders(datasets)

    # Invalidate image collections by id
    if len(image_collections) > 0:
        logging.getLogger(__name__).info("Invalidating image collection {0}".format(image_collections))
        invalidate.invalidate_image_collections(image_collections)

    # Invalidate trial results
    if len(trial_results) > 0:
        logging.getLogger(__name__).info("Invalidating trial result {0}".format(trial_results))
        invalidate.invalidate_trial_results(trial_results)

    # Invalidate metrics
    if len(metrics) > 0:
        logging.getLogger(__name__).info("Invalidating benchmark {0}".format(metrics))
        invalidate.invalidate_metrics(metrics)

    # Invalidate metric results
    if len(metric_results) > 0:
        logging.getLogger(__name__).info("Invalidating result {0}".format(metric_results))
        invalidate.invalidate_metric_results(metric_results)

    # Invalidate failed trials
    if failed_trials:
        logging.getLogger(__name__).info("Invalidating failed trials")
        invalidate.invalidate_failed_trial_results()

    # Invalidate failed metrics
    if failed_metrics:
        logging.getLogger(__name__).info("Invalidating failed metric results")
        invalidate.invalidate_failed_metric_results()

    # Remove incomplete tasks
    if incomplete_tasks:
        logging.getLogger(__name__).info("Removing incomplete tasks")
        invalidate.invalidate_incomplete_tasks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Invalidate obsolete or incorrect data from the database.\n'
                    'Over time, data in the database will become invalid, due to new code versions, '
                    'bugs in the original implementation, or changed input. '
                    'When that happens, use this script to clear out the invalid data while retaining '
                    'other important data')
    parser.add_argument('--system', action='append',
                        help='A system id or type to invalidate. Can be a database id or a fully-qualified class')
    parser.add_argument('--dataset', action='append',
                        help='A dataset loader module to invalidate, a fully-qualified loader module name')
    parser.add_argument('--image_collection', action='append',
                        help='An image collection to invalidate, as a bson database id')
    parser.add_argument('--simulator', action='append',
                        help='An simulator to invalidate, as a bson database id or world name')
    parser.add_argument('--controller', action='append',
                        help='An controller to invalidate, as a bson database id')
    parser.add_argument('--trial_result', action='append',
                        help='A trial result to invalidate, as a bson database id')
    parser.add_argument('--metric', action='append',
                        help='A mtric to invalidate, as a bson database id')
    parser.add_argument('--metric_result', action='append',
                        help='An metric result to invalidate, as a bson database id')
    parser.add_argument('--failed_trials', action='store_true',
                        help='Also remove all failed trial results')
    parser.add_argument('--failed_metrics', action='store_true',
                        help='Also remove all failed metric results')
    parser.add_argument('--incomplete_tasks', action='store_true',
                        help='Remove all incomplete tasks, the necessary ones will be recreated by scheduling')
    parser.add_argument('--mongodb_host', default=None,
                        help='Override the mongodb hostname specified in the config file.')
    parser.add_argument('--mongodb_port', default=None,
                        help='Override the mongodb port specified in the config file.')
    args = parser.parse_args()

    main(
        systems=args.system,
        datasets=args.dataset,
        image_collections=args.image_collection,
        # simulators=args.simulator,
        # controllers=args.controller,
        trial_results=args.trial_result,
        metrics=args.metric,
        metric_results=args.metric_result,
        failed_trials=args.failed_trials,
        failed_metrics=args.failed_metrics,
        incomplete_tasks=args.incomplete_tasks,
        mongodb_host=args.mongodb_host,
        mongodb_port=args.mongodb_port
    )
