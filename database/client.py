from pymongo import MongoClient
from database.entity import Entity
from util.dict_utils import defaults


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
                'connection_parameters': <kwargs passed to MongoClient>
                'database_name': <database name>
                'collections': {
                    'dataset_collection': <collection name for datasets>
                    'images_collection': <collection name for images>
                    'trials_collection': <collection name for trial results>
                    'results_collection': <collection name for results>
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
        db_config = defaults(db_config, {
            'connection_parameters': {},
            'database_name': 'benchmark_system',
            'collections': {
                'dataset_collection': 'datasets',
                'image_collection':  'images',
                'trials_collection': 'trials',
                'results_collection': 'results',
                'trained_state_collection': 'trained_states'
            }
        })

        conn_kwargs = db_config['connection_parameters']
        db_name = db_config['database_name']
        self._dataset_collection_name = db_config['collections']['dataset_collection']
        self._image_collection_name = db_config['collections']['image_collection']
        self._trials_collection_name = db_config['collections']['trials_collection']
        self._results_collection_name = db_config['collections']['results_collection']
        self._trained_state_collection_name = db_config['collections']['trained_state_collection']

        self._mongo_client = MongoClient(**conn_kwargs)
        self._database = self._mongo_client[db_name]

        self._entity_register = {}

    @property
    def dataset_collection(self):
        return self._database[self._dataset_collection_name]

    @property
    def image_collection(self):
        return self._database[self._image_collection_name]

    @property
    def trials_collection(self):
        return self._database[self._trials_collection_name]

    @property
    def results_collection(self):
        return self._database[self._results_collection_name]

    @property
    def trained_state_collection(self):
        return self._database[self._trained_state_collection_name]

    def register_entity(self, entity_class):
        if issubclass(entity_class, Entity):
            self._entity_register[entity_class.__name__] = entity_class

    def deserialize_entity(self, s_entity):
        type_name = s_entity['_type']
        if type_name in self._entity_register:
            return self._entity_register[type_name].deserialize(s_entity)
        return None
