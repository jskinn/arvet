#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
import bson
import typing
import re

import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.batch_analysis.invalidate


def is_id(val: str) -> bool:
    """
    Is a given string a bson id. It should be a string of hex digits
    :param val:
    :return:
    """
    return re.fullmatch('[0-9a-f]+', val) is not None


def main(systems: typing.List[str] = None, datasets: typing.List[str] = None,
         image_collections: typing.List[str] = None,
         simulators: typing.List[str] = None,
         controllers: typing.List[str] = None,
         trial_results: typing.List[str] = None,
         benchmarks: typing.List[str] = None, benchmark_results: typing.List[str] = None,
         failed_trials: bool = True):
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
    if simulators is None:
        simulators = []
    if controllers is None:
        controllers = []
    if trial_results is None:
        trial_results = []
    if benchmarks is None:
        benchmarks = []
    if benchmark_results is None:
        benchmark_results = []

    # Sanitize arguments
    image_collections = [bson.ObjectId(oid) for oid in image_collections if is_id(oid)]
    controllers = [bson.ObjectId(oid) for oid in controllers if is_id(oid)]
    trial_results = [bson.ObjectId(oid) for oid in trial_results if is_id(oid)]
    benchmarks = [bson.ObjectId(oid) for oid in benchmarks if is_id(oid)]
    benchmark_results = [bson.ObjectId(oid) for oid in benchmark_results if is_id(oid)]

    config = global_conf.load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])
    db_client = arvet.database.client.DatabaseClient(config=config)

    # Invalidate systems
    for system_id in systems:
        if is_id(system_id):
            logging.getLogger(__name__).info("Invalidating system {0}".format(system_id))
            arvet.batch_analysis.invalidate.invalidate_system(db_client, bson.ObjectId(system_id))
        else:
            # get a list of all the ids to invalidate before we start modifying the database
            logging.getLogger(__name__).info("Invalidating all systems of type {0}".format(system_id))
            ids_to_remove = [inner_id['_id'] for inner_id in db_client.system_collection.find({
                '_type': system_id
            }, {'_id': True})]
            for inner_id in ids_to_remove:
                arvet.batch_analysis.invalidate.invalidate_system(db_client, inner_id)

    # Invalidate datasets by loader module
    for loader_module in datasets:
        logging.getLogger(__name__).info("Invalidating dataset loader {0}".format(loader_module))
        arvet.batch_analysis.invalidate.invalidate_dataset_loader(db_client, loader_module)

    # Invalidate image collections by id
    for image_collection_id in image_collections:
        logging.getLogger(__name__).info("Invalidating image collection {0}".format(image_collection_id))
        arvet.batch_analysis.invalidate.invalidate_image_collection(db_client, image_collection_id)

    # Invalidate simulators
    for simulator_id in simulators:
        if is_id(simulator_id):
            logging.getLogger(__name__).info("Invalidating simulator {0}".format(simulator_id))
            arvet.batch_analysis.invalidate.invalidate_simulator(db_client, bson.ObjectId(simulator_id))
        else:
            logging.getLogger(__name__).info("Invalidating all simulators with world '{0}'".format(simulator_id))
            ids_to_remove = [inner_id['_id'] for inner_id in db_client.image_source_collection.find({
                'world_name': simulator_id
            }, {'_id': True})]
            for inner_id in ids_to_remove:
                arvet.batch_analysis.invalidate.invalidate_simulator(db_client, inner_id)

    # Invalidate controllers
    for controller_id in controllers:
        logging.getLogger(__name__).info("Invalidating controller {0}".format(controller_id))
        arvet.batch_analysis.invalidate.invalidate_controller(db_client, controller_id)

    # Invalidate trial results
    for trial_result_id in trial_results:
        logging.getLogger(__name__).info("Invalidating trial result {0}".format(trial_result_id))
        arvet.batch_analysis.invalidate.invalidate_trial_result(db_client, trial_result_id)

    # Invalidate benchmarks
    for benchmark_id in benchmarks:
        logging.getLogger(__name__).info("Invalidating benchmark {0}".format(benchmark_id))
        arvet.batch_analysis.invalidate.invalidate_benchmark(db_client, benchmark_id)

    # Invalidate benchmarks
    for result_id in benchmark_results:
        logging.getLogger(__name__).info("Invalidating result {0}".format(result_id))
        arvet.batch_analysis.invalidate.invalidate_benchmark_result(db_client, result_id)

    # Invalidate failed trials
    if failed_trials:
        failed_trials = db_client.trials_collection.find({'success': False}, {'_id': True})
        for trial_id in failed_trials:
            logging.getLogger(__name__).info("Invalidating failed trial {0}".format(trial_id['_id']))
            arvet.batch_analysis.invalidate.invalidate_trial_result(db_client, trial_id['_id'])


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
    parser.add_argument('--benchmark', action='append',
                        help='A benchmark to invalidate, as a bson database id')
    parser.add_argument('--benchmark_result', action='append',
                        help='An benchmark result to invalidate, as a bson database id')
    parser.add_argument('--failed_trials', action='store_true',
                        help='Also remove all failed trial results')
    args = parser.parse_args()

    main(
        systems=args.system,
        datasets=args.dataset,
        image_collections=args.image_collection,
        simulators=args.simulator,
        controllers=args.controller,
        trial_results=args.trial_result,
        benchmarks=args.benchmark,
        benchmark_results=args.benchmark_result,
        failed_trials=args.failed_trials
    )
