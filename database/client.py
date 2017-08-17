import os
import sys
import logging
import traceback
import pymongo
import gridfs
import importlib
import database.entity
import database.entity_registry
import util.dict_utils as du


class DatabaseClient:
    """
    A wrapper class for maintaining a mongodb database connection.
    Handles the configuration for connecting to databases,
    and which collections should which entities be stored in.
    Use the property accessors to get the appropriate collection for
    each entity type.
    """

    def __init__(self, config=None):
        """
        Construct a database connection
        Takes configuration parameters in a dict with the following format:
        {
            'database_config': {
                'connection_parameters': <kwargs passed to MongoClient>,
                'database_name': <database name>,
                'gridfs_bucket': <gridfs bucket name>,
                'temp_folder': <folder to store temporary files>,
                'collections': {
                    'trainer_collection': <collection name for trainers>
                    'trainee_collection': <collection name for trainees>
                    'system_collection': <collection name for systems>
                    'image_source_collection': <collection name for image sources>
                    'image_collection':  <collection name for images>
                    'trials_collection': <collection name for trial results>
                    'benchmarks_collection': <collection name for benchmarks>
                    'results_collection': <collection name for benchmark results>
                    'experiments_collection': <collection name for experiments>
                    'tasks_collection': <collection name for tasks>
                }
            }
        }
        :param config: Configuration parameters, to set different properties for interacting with MongoDB.
        """
        # configuration keys, to avoid misspellings
        if config is not None and 'database_config' in config:
            db_config = config['database_config']
        else:
            db_config = {}

        # Default configuration. Also serves as an exemplar configuration argument
        db_config = du.defaults(db_config, {
            'connection_parameters': {},
            'database_name': 'benchmark_system',
            'gridfs_bucket': 'fs',
            'temp_folder': 'temp',
            'collections': {
                'trainer_collection': 'trainers',
                'trainee_collection': 'trainees',
                'system_collection': 'systems',
                'image_source_collection': 'image_sources',
                'image_collection':  'images',
                'trials_collection': 'trials',
                'benchmarks_collection': 'benchmarks',
                'results_collection': 'results',
                'experiments_collection': 'experiments',
                'tasks_collection': 'tasks'
            }
        }, modify_base=False)

        conn_kwargs = db_config['connection_parameters']
        db_name = db_config['database_name']
        self._temp_folder = db_config['temp_folder']
        os.makedirs(self._temp_folder, exist_ok=True)   # Make sure the temp folder exists.
        self._trainer_collection_name = db_config['collections']['trainer_collection']
        self._trainee_collection_name = db_config['collections']['trainee_collection']
        self._system_collection_name = db_config['collections']['system_collection']
        self._image_source_collection_name = db_config['collections']['image_source_collection']
        self._image_collection_name = db_config['collections']['image_collection']
        self._trials_collection_name = db_config['collections']['trials_collection']
        self._benchmarks_collection_name = db_config['collections']['benchmarks_collection']
        self._results_collection_name = db_config['collections']['results_collection']
        self._experiments_collection_name = db_config['collections']['experiments_collection']
        self._tasks_collection_name = db_config['collections']['tasks_collection']

        self._mongo_client = pymongo.MongoClient(**conn_kwargs)
        self._database = self._mongo_client[db_name]
        self._gridfs = gridfs.GridFS(self._database, collection=db_config['gridfs_bucket'])

    @property
    def trainer_collection(self):
        return self._database[self._trainer_collection_name]

    @property
    def trainee_collection(self):
        return self._database[self._trainee_collection_name]

    @property
    def system_collection(self):
        return self._database[self._system_collection_name]

    @property
    def image_source_collection(self):
        return self._database[self._image_source_collection_name]

    @property
    def image_collection(self):
        return self._database[self._image_collection_name]

    @property
    def trials_collection(self):
        return self._database[self._trials_collection_name]

    @property
    def benchmarks_collection(self):
        return self._database[self._benchmarks_collection_name]

    @property
    def results_collection(self):
        return self._database[self._results_collection_name]

    @property
    def experiments_collection(self):
        return self._database[self._experiments_collection_name]

    @property
    def tasks_collection(self):
        return self._database[self._tasks_collection_name]

    @property
    def grid_fs(self):
        return self._gridfs

    @property
    def temp_folder(self):
        return self._temp_folder

    def deserialize_entity(self, s_entity):
        type_name = s_entity['_type']
        module_ = type_name.rpartition('.')[0]
        if module_ is not '' and module_ not in sys.modules:
            try:
                importlib.import_module(module_)
            except ImportError:
                logging.getLogger(__name__).error("Could not import module {0} containing type {1}:\{2}".format(
                    module_, type_name, traceback.format_exc()))
                pass
        entity_type = database.entity_registry.get_entity_type(type_name)
        if entity_type:
            return entity_type.deserialize(s_entity, self)
        raise ValueError("Could not deserialize entity type: {0}, make sure it's imported".format(type_name))
