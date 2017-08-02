import sys
import os.path
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
        db_client = database.client.DatabaseClient(config=config)

        # Try and import the desired loader module
        try:
            loader_module = importlib.import_module(loader_module_name)
        except ImportError:
            loader_module = None
        if loader_module is None or not hasattr(loader_module, 'import_dataset'):
            return

        # It's up to the importer to fail here if the path doesn't exist
        dataset_id = loader_module.import_dataset(path, db_client)

        if dataset_id is not None:
            experiment = dh.load_object(db_client, db_client.experiments_collection, experiment_id)
            if experiment is not None:
                experiment.add_image_source(dataset_id, path, db_client)


if __name__ == '__main__':
    main(*sys.argv[1:])
