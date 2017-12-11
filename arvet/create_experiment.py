# Copyright (c) 2017, John Skinner
import typing
import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.batch_analysis.experiment as ex


def create_experiment(experiment: typing.Union[ex.Experiment, typing.Type[ex.Experiment]]) -> None:
    """
    A helper to create experiments within the database.
    :param experiment: An experiment type or object
    :return: void
    """
    config = global_conf.load_global_config('config.yml')
    db_client = arvet.database.client.DatabaseClient(config=config)
    ex.create_experiment(db_client, experiment)
