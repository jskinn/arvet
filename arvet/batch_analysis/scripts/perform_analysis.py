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
    Run analysis for a particular experiment
    :args: Only argument is the id of the task to run
    :return:
    """
    if len(args) >= 1:
        experiment_id = bson.objectid.ObjectId(args[0])

        config = global_conf.load_global_config('config.yml')
        if __name__ == '__main__':
            # Only configure the logging if this is the main function, don't reconfigure
            logging.config.dictConfig(config['logging'])
        db_client = arvet.database.client.DatabaseClient(config=config)

        experiment = dh.load_object(db_client, db_client.experiments_collection, experiment_id)
        if experiment is not None:
            try:
                experiment.perform_analysis(db_client)
            except Exception as exception:
                logging.getLogger(__name__).error("Exception occurred while {0} was performing analysis: {1}".format(
                    type(experiment).__name__, traceback.format_exc()
                ))
                raise exception


if __name__ == '__main__':
    main(*sys.argv[1:])
