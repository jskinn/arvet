# Copyright (c) 2017, John Skinner
import os.path
import time
import random
import logging
import yaml
try:
    from yaml import CDumper as YamlDumper, CLoader as YamlLoader
except ImportError:
    from yaml import Dumper as YamlDumper, Loader as YamlLoader

import arvet.util.dict_utils as du


def load_global_config(filename):
    # Default global configuration.
    # This is what you get if you don't have a configuration file.
    config = {
        'paths': ['~'],
        'database_config': {    # Copied from arvet.database.config. Keep them in sync
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
                'experiments_collection': 'experiments',
                'tasks_collection': 'tasks'
            }
        },
        'task_config': {
            'allow_generate_dataset': True,
            'allow_import_dataset': True,
            'allow_train_system': True,
            'allow_run_system': True,
            'allow_benchmark': True,
            'allow_trial_comparison': True,
            'allow_benchmark_comparison': True,
            'allow_experiment_analysis': True
        },
        'job_system_config': {
            'job_system': 'simple',
            'node_id': 'unknown-job-system'
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
        loaded_config = None
        repeat = 1
        while loaded_config is None and repeat <= 3:
            with open(filename, 'r') as config_file:
                loaded_config = yaml.load(config_file, YamlLoader)
            if loaded_config is None:
                # Failed to load the config file for some reason, wait and try again
                time.sleep(random.uniform(1, 3 * repeat))
                repeat += 1
        if loaded_config is not None:
            config = du.defaults(loaded_config, config)
    else:
        # No global configuration file, create a default one
        save_global_config(filename, config)
    return config


def save_global_config(filename, config):
    with open(filename, 'w+') as config_file:
        return yaml.dump(config, config_file, YamlDumper)
