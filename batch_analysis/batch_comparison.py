# Copyright (c) 2017, John Skinner
import copy
import database.client
import core.trial_comparison
import util.dict_utils as du
import util.database_helpers as dbutil


def compare_results(benchmark, database_client, config=None, trained_state_id=None):
    if (not isinstance(benchmark, core.trial_comparison.TrialComparison) or
            not isinstance(database_client, database.client.DatabaseClient)):
        return

    if config is None:
        config = {}
    else:
        config = dict(config)
    config = du.defaults(config, {
    })

    # Get all the reference datasets, ones that have maximum quality
    reference_dataset_ids = database_client.dataset_collection.find({
        'material_properties.RoughnessQuality': 1,
        'material_properties.BaseMipMapBias': 0,
        'material_properties.NormalQuality': 1,
        'geometry_properties.Forced LOD level': 0
    }, {'_id': True})
    reference_dataset_ids = [result['_id'] for result in reference_dataset_ids]

    # Get the reference trial results, as IDs so the cursor doesn't expire
    reference_trial_ids_query = du.defaults(benchmark.get_benchmark_requirements(),
                                            {'success': True, 'dataset': {'$in': reference_dataset_ids}})
    if trained_state_id is not None:
        reference_trial_ids_query['trained_state'] = trained_state_id
    reference_trial_ids = database_client.trials_collection.find(du.defaults(
        benchmark.get_benchmark_requirements(),
        {'success': True, 'dataset': {'$in': reference_dataset_ids}}
    ), {'_id': True})
    reference_trial_ids = [result['_id'] for result in reference_trial_ids]

    # For each reference trial result
    for ref_trial_id in reference_trial_ids:
        s_temp = database_client.trials_collection.find_one({'_id': ref_trial_id})
        reference_trial = database_client.deserialize_entity(s_temp)

        s_temp = database_client.dataset_collection.find_one({'_id': reference_trial.image_source_id})
        reference_dataset = database_client.deserialize_entity(s_temp)
        reference_dataset_images = reference_dataset.load_images(database_client)

        # Find all dataset ids with the same world details, but different quality settings
        comparison_query = dbutil.query_to_dot_notation({
            'world_name': reference_dataset.world_name,
            'world_information': copy.deepcopy(reference_dataset.world_information)
        })
        #comparison_query['_id'] = {'$ne': ref_trial_id}
        comparison_dataset_ids = database_client.dataset_collection.find(comparison_query, {'_id': True})
        comparison_dataset_ids = [result['_id'] for result in comparison_dataset_ids]

        # Existing comparisons
        existing_compared_trials = database_client.results_collection.find({
            'benchmark': benchmark.identifier,
            'reference': reference_trial.identifier
        },{'trial_result': True, '_id': False})
        existing_compared_trials = [val['trial_result'] for val in existing_compared_trials]

        # Find all trials on these comparison datasets by the same system as the reference trial
        s_comparison_trials = database_client.trials_collection.find(du.defaults(
            benchmark.get_benchmark_requirements(),
            {
                '_id': {'$ne': ref_trial_id, '$nin': existing_compared_trials},
                'dataset': {'$in': comparison_dataset_ids},
                'system': reference_trial.system_id,
                'trained_state': reference_trial.trained_state_id
            }))

        for s_comparision_trial in s_comparison_trials:
            comparison_trial = database_client.deserialize_entity(s_comparision_trial)
            existing_count = database_client.results_collection.find({
                'benchmark': benchmark.identifier,
                'trial_result': comparison_trial.identifier,
                'reference': reference_trial.identifier
            }).count()
            if existing_count <= 0:
                benchmark_result = benchmark.compare_trial_results(comparison_trial,
                                                                   reference_trial,
                                                                   reference_dataset_images)
                database_client.results_collection.insert(benchmark_result.serialize())
