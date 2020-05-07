#!/usr/bin/env python3
# Copyright (c) 2020, John Skinner
import logging
import logging.config
import argparse
from pathlib import Path

from arvet.config.global_configuration import load_global_config
from arvet.config.path_manager import PathManager
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.database.autoload_modules import autoload_modules
from arvet.batch_analysis.experiment import Experiment


def main(experiment_id: str = '', output: str = '', mongodb_host: str = None, mongodb_port: int = None):
    """

    :param experiment_id:
    :param output:
    :param mongodb_host:
    :param mongodb_port:
    """
    # Load the configuration
    config = load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])

    # Configure the database and the image manager
    dbconn.configure(config['database'], override_host=mongodb_host, override_port=mongodb_port)
    im_manager.configure(config['image_manager'])

    # Set up the path manager
    path_manager = PathManager(paths=config['paths'], temp_folder=config['temp_folder'],
                               output_dir=config.get('output_dir', None))

    # No experiment specified, just list the options
    if experiment_id is None or experiment_id == '':
        print("Which experiment would you like to plot?")
        return print_available_experiments()

    # Get the experiment, pre-loading the type
    autoload_modules(Experiment, [experiment_id])
    try:
        experiment = Experiment.objects.get({'_id': experiment_id})
    except Experiment.DoesNotExist:
        # Could not find experiment, print the list of valid ones, and return
        print("Could not find experiment \"{0}\"".format(experiment_id))
        return print_available_experiments()

    # Find an output dir, either using a selected one, or from the path manager
    if output is None or len(output) == 0:
        output_dir = path_manager.get_output_dir()
    else:
        output_dir = Path(output).expanduser().resolve()

    # Delegate to the experiment object to get and store the data
    experiment.cache_plot_data(cache_folder=output_dir)


def print_available_experiments() -> None:
    """
    Print out the list of valid experiment ids. These will be human-readable names
    :return: Nothing
    """
    autoload_modules(Experiment)    # To actually get all the experiments, pymodm needs to know their types.
    print("Valid experiments are:")
    for obj in Experiment.objects.all().only('_id').values():
        print("  {0}".format(obj['_id']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Cache the plot data for a particular experiment.\n'
                    'cache_plot_data.py              -> List the available experiments.\n'
                    'cache_plot_data.py MyExperiment -> Cache the plot data for \"MyExperiment\"\n')
    parser.add_argument('--output', default=None,
                        help='Specify the output location for the cache files.')
    parser.add_argument('--mongodb_host', default=None,
                        help='Override the mongodb hostname specified in the config file.')
    parser.add_argument('--mongodb_port', default=None,
                        help='Override the mongodb port specified in the config file.')
    parser.add_argument('experiment_id', nargs='?', default='',
                        help='The experiment id to plot results for. '
                        'Leave blank to list the available experiments')

    args = parser.parse_args()
    main(
        experiment_id=args.experiment_id,
        output=args.output,
        mongodb_host=args.mongodb_host,
        mongodb_port=args.mongodb_port
    )
