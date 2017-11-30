#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging.config
import argus.database.client
import argus.config.global_configuration as global_conf
import argus.util.database_helpers as dh


def main(*args):
    config = global_conf.load_global_config("config.yml")
    if __name__ == '__main__':
        logging.config.dictConfig(config['logging'])
    db_client = argus.database.client.DatabaseClient(config)
    experiment_ids = db_client.experiments_collection.find({'enabled': {'$ne': False}}, {'_id': True})
    for ex_id in experiment_ids:
        experiment = dh.load_object(db_client, db_client.experiments_collection, ex_id['_id'])
        if experiment is not None and experiment.enabled:
            experiment.plot_results(db_client)


if __name__ == "__main__":
    main()
