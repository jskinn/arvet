#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import traceback
import argparse
from bson import ObjectId

from arvet.config.global_configuration import load_global_config
from arvet.config.path_manager import PathManager
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.database.autoload_modules import autoload_modules
from arvet.batch_analysis.task import Task


def main(task_id: str, config_file: str = 'config.yml', allow_write: bool = False, mongodb_host: str = None, mongodb_port: int = None):
    """
    Run a particular task.
    :args: Only argument is the id of the task to run
    :return:
    """
    task_id = ObjectId(task_id)

    # Load the configuration
    config = load_global_config(config_file)
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])

    # Configure the database and the image manager
    dbconn.configure(config['database'], override_host=mongodb_host, override_port=mongodb_port)
    im_manager.configure(config['image_manager'], allow_write)

    # Set up the path manager
    path_manager = PathManager(paths=config['paths'], temp_folder=config['temp_folder'],
                               output_dir=config.get('output_dir', None))

    # Try and get the task object
    autoload_modules(Task, [task_id])   # Try and autoload the task subclass
    try:
        task = Task.objects.get({'_id': task_id})
    except Exception as ex:
        logging.getLogger(__name__).critical("Exception occurred while loading Task({0}):\n{1}".format(
            str(task_id), traceback.format_exc()
        ))
        raise ex

    # Since we got the task, try and run it
    try:
        task.run_task(path_manager)
    except Exception as ex:
        logging.getLogger(__name__).critical("Exception occurred while running {0}({1}):\n{2}".format(
            type(task).__name__, str(task_id), traceback.format_exc()
        ))
        task.mark_job_failed()
        raise ex
    finally:
        task.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a particular task, by id.'
                    'This should be called automatically by the job system. Common to all task types.')
    parser.add_argument('--config', default='config.yml',
                        help='The path to the config file to use. default to \'config.yml\'')
    parser.add_argument('--mongodb_host', default=None,
                        help='Override the monbodb hostname specified in the config file. '
                             'the job system can use this for dynamic config.')
    parser.add_argument('--mongodb_port', default=None,
                        help='Override the monbodb port specified in the config file. '
                             'The job system may use this for dynamic config.')
    parser.add_argument('--allow_write', action='store_true',
                        help='Configure the image manager to allow writing to the HDF5 file. '
                             'Only one such task may exist at a time.')
    parser.add_argument('task_id',
                        help='The id of the task to run.')

    args = parser.parse_args()
    main(args.task_id, args.config, args.allow_write, args.mongodb_host, args.mongodb_port)
