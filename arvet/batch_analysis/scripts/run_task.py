#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import sys
import os
import logging
import logging.config
import traceback
from bson import ObjectId

from arvet.config.global_configuration import load_global_config
from arvet.config.path_manager import PathManager
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.batch_analysis.task import Task


def patch_cwd():
    """
    Patch sys.path to make sure the current working directory is included.
    This is necessary when this is being used as a library,
    and we run the script by file path.
    :return:
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)


def main(*args):
    """
    Run a particular task.
    :args: Only argument is the id of the task to run
    :return:
    """
    if len(args) >= 1:
        task_id = ObjectId(args[0])
        # patch_cwd()

        # Load the configuration
        config = load_global_config('config.yml')
        if __name__ == '__main__':
            # Only configure the logging if this is the main function, don't reconfigure
            logging.config.dictConfig(config['logging'])

        # Configure the database and the image manager
        dbconn.configure(config['database'])
        im_manager.configure(config['image_manager'])

        # Set up the path manager
        path_manger = PathManager(paths=config['paths'])

        # Try and get the task object
        try:
            task = Task.objects.get({'_id': task_id})
        except Exception as ex:
            logging.getLogger(__name__).critical("Exception occurred while loading Task({0}):\n{1}".format(
                str(task_id), traceback.format_exc()
            ))
            raise ex

        # Since we got the task, try and run it
        try:
            task.run_task(path_manger)
        except Exception as ex:
            logging.getLogger(__name__).critical("Exception occurred while running {0}({1}):\n{2}".format(
                type(task).__name__, str(task_id), traceback.format_exc()
            ))
            task.mark_job_failed()
            raise ex
        finally:
            task.save()


if __name__ == '__main__':
    main(*sys.argv[1:])
