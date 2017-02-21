import os.path
from yaml import dump as yaml_dump, load as yaml_load
try:
    from yaml import CDumper as YamlDumper, CLoader as YamlLoader
except ImportError:
    from yaml import Dumper as YamlDumper, Loader as YamlLoader

import util.dict_utils as du


def load_global_config(filename):
    config = {
        'desired_repeats': 10,
        'database_config': {
            'connection_parameters': {},
            'database_name': 'benchmark_system',
            'collections': {
                'dataset_collection': 'datasets',
                'image_collection': 'images',
                'trials_collection': 'trials',
                'results_collection': 'results',
                'trained_state_collection': 'trained_states'
            }
        }
    }
    if os.path.isfile(filename):
        with open(filename, 'r') as config_file:
            config = du.defaults(yaml_load(config_file, YamlLoader), config)
    return config


def save_global_config(filename, config):
    with open(filename, 'w') as config_file:
        return yaml_dump(config, config_file, YamlDumper)
