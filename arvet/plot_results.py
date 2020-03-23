#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import typing
import argparse
from pathlib import Path

from arvet.config.global_configuration import load_global_config
from arvet.config.path_manager import PathManager
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.database.autoload_modules import autoload_modules
from arvet.batch_analysis.experiment import Experiment


def main(experiment_id: str = '', plot_names: typing.Collection[str] = None, show: bool = True, output: str = '',
         mongodb_host: str = None, mongodb_port: int = None):
    """

    :param experiment_id:
    :param plot_names:
    :param show:
    :param output:
    :param mongodb_host:
    :param mongodb_port:
    :return:
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

    # No plots given, print the list of available plots
    if plot_names is None or len(plot_names) <= 0:
        return print_available_plots(experiment)

    # Find an output dir, either using a selected one, or from the path manager
    if output is None or len(output) == 0:
        output_dir = path_manager.get_output_dir()
    else:
        output_dir = Path(output).expanduser().resolve()

    # Delegate to the experiment object to plot the results
    experiment.plot_results(
        output_dir=output_dir,
        plot_names=plot_names,
        display=show
    )


def print_available_experiments() -> None:
    """
    Print out the list of valid experiment ids. These will be human-readable names
    :return: Nothing
    """
    autoload_modules(Experiment)    # To actually get all the experiments, pymodm needs to know their types.
    print("Valid experiments are:")
    for obj in Experiment.objects.all().only('_id').values():
        print("  {0}".format(obj['_id']))


def print_available_plots(experiment: Experiment):
    """
    Print
    :param experiment:
    :return:
    """
    available_plots = experiment.get_plots()
    if len(available_plots) > 0:
        print("Available plots for {0}:".format(experiment.name))
        for plot_name in available_plots:
            print("  {0}".format(plot_name))
    else:
        print("No plots available for {0}".format(experiment.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot the results from an experiment.\n'
                    'plot_results.py                     -> List the available experiments.\n'
                    'plot_results.py MyExperiment        -> List the available plots for \"MyExperiment\"\n'
                    'plot_results.py MyExperiment MyPlot -> Create plot \"MyPlot\"')
    parser.add_argument('--no-display', action='store_true', dest='hide',
                        help='Don\'t actually show the plots as they are generated. '
                             'By default')
    parser.add_argument('--output', default='',
                        help='Override the output location for the plots.')
    parser.add_argument('--mongodb_host', default=None,
                        help='Override the mongodb hostname specified in the config file.')
    parser.add_argument('--mongodb_port', default=None,
                        help='Override the mongodb port specified in the config file.')
    parser.add_argument('experiment_id', nargs='?', default='',
                        help='The experiment id to plot results for. '
                        'Leave blank to list the available experiments')
    parser.add_argument('plots', metavar='plot_name', nargs='*', default=[],
                        help='The names of the plots to create. '
                        'Omit to print the list of available plots for the given experiment')

    args = parser.parse_args()
    main(args.experiment_id, args.plots, not args.hide, args.output, args.mongodb_host, args.mongodb_port)
