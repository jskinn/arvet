#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
import typing
import traceback
import bson

from arvet.config.global_configuration import load_global_config
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.database.autoload_modules import autoload_modules
import arvet.batch_analysis.task_manager as task_manager
from arvet.batch_analysis.experiment import Experiment
import arvet.batch_analysis.job_systems.job_system_factory as job_system_factory


def schedule(config_file: str, schedule_tasks: bool = True, run_tasks: bool = True,
             experiment_ids: typing.List[str] = None):
    """
    Schedule tasks for all experiments.
    We need to find a way of running this repeatedly as a daemon
    :param config_file: The location of the config file to load.
    :param schedule_tasks: Whether to schedule execution tasks for the experiments. Default true.
    :param experiment_ids: A limited set of experiments to schedule for. Default None, which is all experiments.
    :param run_tasks: Actually use the job system to execute scheduled tasks
    """
    # Load the configuration
    config = load_global_config(config_file)
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])

    # Configure the database and the image manager
    dbconn.configure(config['database'])
    im_manager.configure(config['image_manager'])

    if schedule_tasks:
        # Build the query and load the relevant experiment types
        query = {'enabled': {'$ne': False}}
        if experiment_ids is not None and len(experiment_ids) > 0:
            experiment_ids = [bson.ObjectId(ex_id) for ex_id in experiment_ids]
            if len(experiment_ids) == 1:
                query['_id'] = experiment_ids[0]
            else:
                query['_id'] = {'$in': experiment_ids}

            logging.getLogger(__name__).info("Loading experiment types...")
            autoload_modules(Experiment, experiment_ids)
        else:
            logging.getLogger(__name__).info("Loading experiment types...")
            autoload_modules(Experiment)

        # Schedule the experiments
        logging.getLogger(__name__).info("Scheduling experiments...")
        for experiment in Experiment.objects.raw(query):
            logging.getLogger(__name__).info(" ... experiment {0}".format(experiment.pk))
            try:
                experiment.schedule_tasks()
                experiment.save()
            except Exception as ex:
                logging.getLogger(__name__).error("Exception occurred during scheduling {0}({1}):\n{2}".format(
                    type(experiment).__name__, str(experiment.pk), traceback.format_exc()
                ))
                raise ex
        logging.getLogger(__name__).info("Scheduling complete, there are {0} pending tasks.".format(
            task_manager.count_pending_tasks()
        ))

    if run_tasks:
        logging.getLogger(__name__).info("Running tasks...")
        job_system = job_system_factory.create_job_system(config=config, config_file=config_file)
        task_manager.schedule_tasks(job_system)

        # Actually run the queued jobs.
        job_system.run_queued_jobs()


def main():
    """
    Run the scheduler from the command line.
    Calls scheduler, above.
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Update and schedule tasks from experiments.'
                    'By default, this will update and schedule tasks for all experiments.')
    parser.add_argument('--config', default='config.yml',
                        help='The path to the config file to use. default to \'config.yml\'')
    parser.add_argument('--skip_schedule_tasks', action='store_true',
                        help='Don\'t schedule the execution or evaluation of systems')
    parser.add_argument('--skip_run_tasks', action='store_true',
                        help='Don\'t actually run tasks, only schedule them')
    parser.add_argument('experiment_ids', metavar='experiment_id', nargs='*', default=[],
                        help='Limit the update to only the specified experiment by id. '
                             'You may specify any number of ids.')

    args = parser.parse_args()
    schedule(args.config, not args.skip_schedule_tasks, not args.skip_run_tasks, args.experiment_ids)


if __name__ == '__main__':
    main()
