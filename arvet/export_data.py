#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
import traceback
from bson import ObjectId

from arvet.config.global_configuration import load_global_config
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.batch_analysis.experiment import Experiment


def main(experiment_ids_to_plot=None):
    """
    Allow experiments to dump some data to file. This might be aggregate statistics,
    I'm currently using this for camera trajectories.
    :return:
    """
    # Load the configuration
    config = load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])

    # Configure the database and the image manager
    dbconn.configure(config['database'])
    im_manager.configure(config['image_manager'])

    query = {'enabled': {'$ne': False}}
    if experiment_ids_to_plot is not None and len(experiment_ids_to_plot) > 0:
        query['_id'] = {'$in': [ObjectId(id_) for id_ in experiment_ids_to_plot]}

    for experiment in Experiment.objects.raw(query, {'_id': True}):
        try:
            experiment.export_data()
        except Exception as ex:
            logging.getLogger(__name__).critical("Exception occurred while performing analysis {0}({1}):\n{2}".format(
                type(experiment).__name__, str(experiment.identifer), traceback.format_exc()
            ))
            raise ex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export data from one or more experiments.\n'
                    'By default, this will export data for all enabled experiments, '
                    'pass a experiment id to limit the results.')
    parser.add_argument('experiment_ids', metavar='experiment_id', nargs='*', default=[],
                        help='Limit the update to only the specified experiment by id. '
                             'You may specify any number of ids.')

    args = parser.parse_args()
    main(args.experiment_ids)
