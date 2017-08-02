import config.global_configuration as global_conf
import database.client
import util.database_helpers as db_help

import experiments.podcup.podcup_experiment
import experiments.visual_slam.visual_slam_experiment

import benchmarks.ate.absolute_trajectory_error as bench_ate
import benchmarks.ate.absolute_trajectory_error_comparison as bench_ate_comp
import benchmarks.loop_closure.loop_closure as bench_closure
import benchmarks.matching.match_comparison as bench_match_comp
import benchmarks.rpe.relative_pose_error as bench_rpe
import benchmarks.rpe.relative_pose_error_comparison as bench_rpe_comp
import benchmarks.tracking.tracking_benchmark as bench_track
import benchmarks.tracking.tracking_comparison_benchmark as bench_track_comp


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
