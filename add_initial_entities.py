import config.global_configuration as global_conf
import database.client
import util.database_helpers as db_help

import experiments.visual_slam.visual_slam_experiment


def main():
    """
    Add hard-coded entities to the database.
    Should become only experiments, when that happens, rename to 'create_experiments.py'
    :return: void
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)

    # Create the experiments
    c = db_client.experiments_collection
    db_help.add_unique(c, experiments.visual_slam.visual_slam_experiment.VisualSlamExperiment())


if __name__ == '__main__':
    main()
