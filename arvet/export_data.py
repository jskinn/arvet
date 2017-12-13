#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.util.database_helpers as dh


def main():
    """
    Allow experiments to dump some data to file. This might be aggregate statistics,
    I'm currently using this for camera trajectories.
    :return:
    """
    config = global_conf.load_global_config('config.yml')
    if __name__ == '__main__':
        logging.config.dictConfig(config['logging'])
    db_client = arvet.database.client.DatabaseClient(config=config)
    experiment_ids = db_client.experiments_collection.find({'enabled': {'$ne': False}}, {'_id': True})
    for ex_id in experiment_ids:
        try:
            experiment = dh.load_object(db_client, db_client.experiments_collection, ex_id['_id'])
        except ValueError:
            # Cannot deserialize experiment, skip to the next one.
            continue

        if experiment is not None and experiment.enabled:
            experiment.export_data(db_client)


if __name__ == '__main__':
    main()
