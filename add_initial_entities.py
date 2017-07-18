import config.global_configuration as global_conf
import database.client
import util.database_helpers as db_help

import experiments.podcup.podcup_experiment

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

    ############
    # Add benchmarks
    # TODO: Move these into individual experiments, then we just have to make the experiments
    ###########
    c = db_client.benchmarks_collection
    db_help.add_unique(c, bench_ate.BenchmarkATE(offset=0, max_difference=0.02, scale=1))
    db_help.add_unique(c, bench_ate_comp.ATEBenchmarkComparison(offset=0, max_difference=0.02))
    db_help.add_unique(c, bench_closure.BenchmarkLoopClosure(distance_threshold=5, trivial_closure_index_distance=2))
    db_help.add_unique(c, bench_match_comp.BenchmarkMatchingComparison(offset=0, max_difference=0.02))
    db_help.add_unique(c, bench_rpe.BenchmarkRPE(max_pairs=10000, fixed_delta=False, delta=1.0,
                                         delta_unit='s', offset=0, scale_=1))
    db_help.add_unique(c, bench_rpe_comp.RPEBenchmarkComparison(offset=0, max_difference=0.02))
    db_help.add_unique(c, bench_track.TrackingBenchmark(initializing_is_lost=True))
    db_help.add_unique(c, bench_track_comp.TrackingComparisonBenchmark(offset=0, max_difference=0.02))

    # Create the experiments
    c = db_client.experiments_collection
    db_help.add_unique(c, experiments.podcup.podcup_experiment.PodCupExperiment())


if __name__ == '__main__':
    main()
