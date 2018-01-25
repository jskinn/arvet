#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging.config
import argparse
import bson
import arvet.database.client
import arvet.config.global_configuration as global_conf
import arvet.util.database_helpers as dh


def main(experiment_ids_to_plot=None):
    config = global_conf.load_global_config("config.yml")
    if __name__ == '__main__':
        logging.config.dictConfig(config['logging'])
    db_client = arvet.database.client.DatabaseClient(config)

    query = {'enabled': {'$ne': False}}
    if experiment_ids_to_plot is not None and len(experiment_ids_to_plot) > 0:
        query['_id'] = {'$in': [bson.ObjectId(id_) for id_ in experiment_ids_to_plot]}
    experiment_ids = db_client.experiments_collection.find(query, {'_id': True})

    for ex_id in experiment_ids:
        try:
            experiment = dh.load_object(db_client, db_client.experiments_collection, ex_id['_id'])
        except ValueError:
            # Cannot deserialize experiment, skip to the next one.
            logging.getLogger(__name__).info("... Cannot deserialize experiment {0} skipping".format(ex_id['_id']))
            continue

        if experiment is not None and experiment.enabled:
            experiment.plot_results(db_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot the results from one or more experiments.\n'
                    'By default, this will plot results for all enabled experiments, '
                    'pass a experiment id to limit the results.')
    parser.add_argument('experiment_ids', metavar='experiment_id', nargs='*', default=[],
                        help='Limit the update to only the specified experiment by id. '
                             'You may specify any number of ids.')

    args = parser.parse_args()
    main(args.experiment_ids)
