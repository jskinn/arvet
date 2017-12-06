#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import sys
import logging
import logging.config

import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.batch_analysis.invalidate


def main():
    """
    Run a particular task.
    :args: Only argument is the id of the task to run
    :return:
    """
    config = global_conf.load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])
    db_client = arvet.database.client.DatabaseClient(config=config)

    orbslam_ids = db_client.system_collection.find({'_type': 'systems.slam.orbslam2.ORBSLAM2'}, {'_id': True})
    for system_id in orbslam_ids:
        logging.getLogger(__name__).info("Invalidating system {0}".format(system_id['_id']))
        arvet.batch_analysis.invalidate.invalidate_system(db_client, system_id['_id'])

    failed_trials = db_client.trials_collection.find({'success': False}, {'_id': True})
    for trial_id in failed_trials:
        logging.getLogger(__name__).info("Invalidating failed trial {0}".format(trial_id['_id']))
        arvet.batch_analysis.invalidate.invalidate_trial_result(db_client, trial_id['_id'])


if __name__ == '__main__':
    main()
