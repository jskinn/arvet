#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import sys
import logging
import logging.config
import traceback
import bson.objectid

import arvet.config.global_configuration as global_conf
import arvet.config.path_manager
import arvet.database.client
import arvet.util.database_helpers as dh


def main(*args):
    """
    Run a particular task.
    :args: Only argument is the id of the task to run
    :return:
    """
    if len(args) >= 1:
        task_id = bson.objectid.ObjectId(args[0])

        config = global_conf.load_global_config('config.yml')
        if __name__ == '__main__':
            # Only configure the logging if this is the main function, don't reconfigure
            logging.config.dictConfig(config['logging'])
        path_manger = arvet.config.path_manager.PathManager(paths=config['paths'])
        db_client = arvet.database.client.DatabaseClient(config=config)

        task = dh.load_object(db_client, db_client.tasks_collection, task_id)
        if task is not None:
            try:
                task.run_task(db_client)
            except Exception:
                logging.getLogger(__name__).error("Exception occurred while running {0}: {1}".format(
                    type(task).__name__, traceback.format_exc()
                ))
                task.mark_job_failed()
            task.save_updates(db_client.tasks_collection)


if __name__ == '__main__':
    main(*sys.argv[1:])
