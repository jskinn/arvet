#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
import typing
import traceback
import bson
import config.global_configuration as global_conf
import database.client
import util.database_helpers as dh
import batch_analysis.task_manager
import batch_analysis.job_systems.job_system_factory as job_system_factory


def main(do_imports: bool = True, schedule_tasks: bool = True, run_tasks: bool = True,
         experiment_ids: typing.List[str] = None):
    """
    Schedule tasks for all experiments.
    We need to find a way of running this repeatedly as a daemon
    :param do_imports: Whether to do imports for all the experiments. Default true.
    :param schedule_tasks: Whether to schedule execution tasks for the experiments. Default true.
    :param experiment_ids: A limited set of experiments to schedule for. Default None, which is all experiments.
    :param run_tasks: Actually use the job system to execute scheduled tasks
    """
    config = global_conf.load_global_config('config.yml')
    if __name__ == '__main__':
        logging.config.dictConfig(config['logging'])
    db_client = database.client.DatabaseClient(config=config)
    task_manager = batch_analysis.task_manager.TaskManager(db_client.tasks_collection, db_client, config)

    if do_imports or schedule_tasks:
        query = {'enabled': {'$ne': False}}
        if experiment_ids is not None and len(experiment_ids) > 0:
            query['_id'] = {'$in': [bson.ObjectId(id_) for id_ in experiment_ids]}
        experiment_ids = db_client.experiments_collection.find(query, {'_id': True})

        logging.getLogger(__name__).info("Scheduling experiments...")
        for experiment_id in experiment_ids:
            experiment = dh.load_object(db_client, db_client.experiments_collection, experiment_id['_id'])
            if experiment is not None and experiment.enabled:
                logging.getLogger(__name__).info(" ... experiment {0}".format(experiment.identifier))
                try:
                    if do_imports:
                        experiment.do_imports(task_manager, db_client)
                    if schedule_tasks:
                        experiment.schedule_tasks(task_manager, db_client)
                    experiment.save_updates(db_client)
                except Exception:
                    logging.getLogger(__name__).error(
                        "Exception occurred during scheduling:\n{0}".format(traceback.format_exc()))

    if run_tasks:
        logging.getLogger(__name__).info("Running tasks...")
        job_system = job_system_factory.create_job_system(config=config)
        task_manager.schedule_tasks(job_system)

        # Actually run the queued jobs.
        job_system.run_queued_jobs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update and schedule tasks from experiments.'
                    'By default, this will update and schedule tasks for all experiments.')
    parser.add_argument('--skip_imports', action='store_true',
                        help='Skip import data for the experiments.')
    parser.add_argument('--skip_schedule_tasks', action='store_true',
                        help='Don\'t schedule the execution or evaluation of systems')
    parser.add_argument('--skip_run_tasks', action='store_true',
                        help='Don\'t actually run tasks, only schedule them')
    parser.add_argument('experiment_ids', metavar='experiment_id', nargs='*', default=[],
                        help='Limit the update to only the specified experiment by id. '
                             'You may specify any number of ids.')

    args = parser.parse_args()
    main(not args.skip_imports, not args.skip_schedule_tasks, not args.skip_run_tasks, args.experiment_ids)
