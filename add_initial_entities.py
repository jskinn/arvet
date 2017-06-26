import config.global_configuration as global_conf
import database.client
import util.database_helpers as db_help
import benchmarks.ate.absolute_trajectory_error as bench_ate
import benchmarks.ate.absolute_trajectory_error_comparison as bench_ate_comp
import benchmarks.loop_closure.loop_closure as bench_closure
import benchmarks.matching.match_comparison as bench_match_comp
import benchmarks.rpe.relative_pose_error as bench_rpe
import benchmarks.rpe.relative_pose_error_comparison as bench_rpe_comp
import benchmarks.tracking.tracking_benchmark as bench_track
import benchmarks.tracking.tracking_comparison_benchmark as bench_track_comp
import benchmarks.bounding_box_overlap.bounding_box_overlap as bench_bbox_overlap


def add_unique(collection, entity):
    """
    Add an object to a collection, if that object does not already exist.
    Treats the entire serialized object as the key, if only one entry is different, they're different objects.
    :param collection: The mongodb collection to insert into
    :param entity: The object to insert
    :return: 
    """
    s_object = entity.serialize()
    if collection.find_one(db_help.query_to_dot_notation(s_object)) is None:
        collection.insert(s_object)


def main():
    """
    Add hard-coded entities to the database.
    Systems with only a single collection of settings,
    benchmarks with no settings, and image sources,
    all of these are added here
    :return: void
    """
    config = global_conf.load_global_config('config.yml')
    db_client = database.client.DatabaseClient(config=config)

    ############
    #  Add benchmarks
    ###########
    c = db_client.benchmarks_collection
    add_unique(c, bench_ate.BenchmarkATE(offset=0, max_difference=0.02, scale=1))
    add_unique(c, bench_ate_comp.ATEBenchmarkComparison(offset=0, max_difference=0.02))
    add_unique(c, bench_closure.BenchmarkLoopClosure(distance_threshold=5, trivial_closure_index_distance=2))
    add_unique(c, bench_match_comp.BenchmarkMatchingComparison(offset=0, max_difference=0.02))
    add_unique(c, bench_rpe.BenchmarkRPE(max_pairs=10000, fixed_delta=False, delta=1.0,
                                         delta_unit='s', offset=0, scale_=1))
    add_unique(c, bench_rpe_comp.RPEBenchmarkComparison(offset=0, max_difference=0.02))
    add_unique(c, bench_track.TrackingBenchmark(initializing_is_lost=True))
    add_unique(c, bench_track_comp.TrackingComparisonBenchmark(offset=0, max_difference=0.02))
    add_unique(c, bench_bbox_overlap.BoundingBoxOverlapBenchmark())


if __name__ == '__main__':
    main()
