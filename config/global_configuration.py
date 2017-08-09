import os.path
import logging
from yaml import dump as yaml_dump, load as yaml_load
try:
    from yaml import CDumper as YamlDumper, CLoader as YamlLoader
except ImportError:
    from yaml import Dumper as YamlDumper, Loader as YamlLoader

import util.dict_utils as du


def load_global_config(filename):
    # Default global configuration.
    # This is what you get if you don't have a configuration file.
    config = {
        'database_config': {    # Copied from database.config. Keep them in sync
            'connection_parameters': {},
            'database_name': 'benchmark_system',
            'gridfs_bucket': 'fs',
            'temp_folder': 'temp',
            'collections': {
                'trainer_collection': 'trainers',
                'trainee_collection': 'trainees',
                'system_collection': 'systems',
                'image_source_collection': 'image_sources',
                'image_collection': 'images',
                'trials_collection': 'trials',
                'benchmarks_collection': 'benchmarks',
                'results_collection': 'results',
                'experiments_collection': 'experiments'
            }
        },
        'job_system_config': {
            'job_system': 'simple'
        },
        'logging': {    # Default logging configuration
            'version': 1,
            'formatters': {
                'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
            },
            'handlers': {
                'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}
            },
            'root': {'handlers': ['h'], 'level': logging.DEBUG},
        }
    }
    if os.path.isfile(filename):
        with open(filename, 'r') as config_file:
            config = du.defaults(yaml_load(config_file, YamlLoader), config)
    save_global_config(filename, config)
    return config


def save_global_config(filename, config):
    with open(filename, 'w+') as config_file:
        return yaml_dump(config, config_file, YamlDumper)
