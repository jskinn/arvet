import os.path
import glob
import pickle
import copy

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

import systems.deep_learning.keras_frcnn as sys_frcnn


def add_unique(collection, entity):
    """
    Add an object to a collection, if that object does not already exist.
    Treats the entire serialized object as the key, if only one entry is different, they're different objects.
    :param collection: The mongodb collection to insert into
    :param entity: The object to insert
    :return: 
    """
    s_object = entity.serialize()
    query = db_help.query_to_dot_notation(copy.deepcopy(s_object))
    if collection.find_one(query) is None:
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
    # Add benchmarks
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

    ###########
    # Add keras frcnns
    ###########
    c = db_client.system_collection
    model_dir = os.path.expanduser('~/keras-models')
    for config_pickle_path in glob.iglob(os.path.join(model_dir, '*.pickle')):
        model_hdf5_path = os.path.splitext(config_pickle_path)[0] + '.hdf5'
        if os.path.isfile(model_hdf5_path):
            with open(config_pickle_path, 'rb') as config_file:
                frcnn_config = pickle.load(config_file)
            frcnn_config.model_path = model_hdf5_path        # Update the path to the model file
            add_unique(c, sys_frcnn.KerasFRCNN(frcnn_config))


if __name__ == '__main__':
    main()
