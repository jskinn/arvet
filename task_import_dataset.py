#Copyright (c) 2017, John Skinner
import sys
import traceback
import logging
import logging.config
import importlib
import bson.objectid

import config.global_configuration as global_conf
import database.client
import util.database_helpers as dh


def main(*args):
    """
    Import a dataset into the database from a given folder.
    :args: First argument is the module containing the do_import function to use,
    second argument is the location of the dataset, (either some file or root folder)
    Third argument is optionally and experiment to give the imported dataset id to.
    :return:
    """
    if len(args) >= 2:
        loader_module_name = str(args[0])
        path = str(args[1])
        experiment_id = bson.objectid.ObjectId(args[2]) if len(args) >= 3 else None

        config = global_conf.load_global_config('config.yml')
        logging.config.dictConfig(config['logging'])
        log = logging.getLogger(__name__)
        db_client = database.client.DatabaseClient(config=config)

        # Try and import the desired loader module
        try:
            loader_module = importlib.import_module(loader_module_name)
        except ImportError:
            loader_module = None
        if loader_module is None:
            log.error("Could not load module {0} for importing dataset, check it  exists".format(loader_module_name))
            return
        if not hasattr(loader_module, 'import_dataset'):
            log.error("Module {0} does not have method 'import_dataset'".format(loader_module_name))
            return

        # It's up to the importer to fail here if the path doesn't exist
        if experiment_id is not None:
            log.info("Importing dataset from {0} using module {1} for experiment {2}".format(path, loader_module_name,
                                                                                             experiment_id))
        else:
            log.info("Importing dataset from {0} using module {1}".format(path, loader_module_name))
        try:
            dataset_id = loader_module.import_dataset(path, db_client)
        except Exception:
            dataset_id = None
            log.error("Exception occurred while importing dataset from {0} with module {1}:\n{2}".format(
                path, loader_module_name, traceback.format_exc()
            ))

        if dataset_id is not None:
            experiment = dh.load_object(db_client, db_client.experiments_collection, experiment_id)
            if experiment is not None:
                log.info("Successfully imported dataset {0}, adding to experiment {1}".format(dataset_id,
                                                                                              experiment_id))
                experiment.add_image_source(dataset_id, path, db_client)
            else:
                log.info("Successfully imported dataset {0}".format(dataset_id))


if __name__ == '__main__':
    main(*sys.argv[1:])
