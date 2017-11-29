# Copyright (c) 2017, John Skinner
import database.client
import core.benchmark
import util.dict_utils as du


def benchmark_results(benchmark, database_client, config=None, trained_state_id=None):
    if (not isinstance(benchmark, core.benchmark.Benchmark) or
            not isinstance(database_client, database.client.DatabaseClient)):
        return

    if config is None:
        config = {}
    else:
        config = dict(config)
    config = du.defaults(config, {

    })

    existing_results_query = {'benchmark': benchmark.identifier}
    if trained_state_id is not None:
        existing_results_query['trained_state'] = trained_state_id
    existing_results = database_client.results_collection.find(existing_results_query,
                                                               {'_id': False, 'trial_result': True})
    existing_results = [result['trial_result'] for result in existing_results]
    trial_results = database_client.trials_collection.find(du.defaults(benchmark.get_benchmark_requirements(),
                                                                       {'_id': {'$nin': existing_results} }))

    for s_trial_result in trial_results:
        trial_result = database_client.deserialize_entity(s_trial_result)
        s_dataset = database_client.dataset_collection.find_one({'_id': trial_result.image_source_id})
        dataset = database_client.deserialize_entity(s_dataset)
        dataset_images = dataset.load_images(database_client)
        benchmark_result = benchmark.benchmark_results(dataset_images, trial_result)
        database_client.results_collection.insert(benchmark_result.serialize())
