# Copyright (c) 2017, John Skinner
import database.client
import core.system
import core.trained_system
import util.dict_utils as du


def test_vision_system(system, database_client, config=None):
    if config is None:
        config = {}
    else:
        config = dict(config)
    # Default configuration
    config = du.defaults({
        'desired_repeats': 10
    }, config)

    # load and cache all the dataset ids, to avoid cursor timeout
    dataset_ids = database_client.dataset_collection.find(system.get_dataset_criteria(), {'_id': True})
    dataset_ids = list(dataset_ids)

    for dataset_id in dataset_ids:
        s_dataset = database_client.dataset_collection.find_one({'_id': dataset_id['_id']})
        dataset = database_client.deserialize_entity(s_dataset)
        dataset_images = dataset.load_images(database_client)
        if len(dataset_images) <= 0:
            raise RuntimeError("Could not load dataset. Did you forget to mount a file share?")

        # get the number of existing trials, so that we don't repeat too many times
        # Some special cases for systems with trained states, so that trials with a different training set don't count
        existing_trials_criteria = {
            'dataset': dataset.identifier,
            'system': system.identifier
        }
        if isinstance(system, core.trained_system.TrainedVisionSystem):
            existing_trials_criteria['trained_state'] = system.trained_state
        num_existing_trials = database_client.trials_collection.find(existing_trials_criteria).count()

        repeats = 1
        if not system.is_deterministic:
            repeats = config['desired_repeats']
        repeats -= num_existing_trials

        for repeat in range(0, repeats):
            trial_result = system.run_with_dataset(dataset_images)
            database_client.trials_collection.insert(trial_result.serialize())


def test_trained_vision_system(system, database_client, config=None):
    if (not isinstance(system, core.trained_system.TrainedVisionSystem) or
            not isinstance(database_client, database.client.DatabaseClient)):
        return

    # Load and store all the state ids, to get around the cursor timing out.
    trained_state_ids = database_client.trained_state_collection.find({'system': system.identifier}, {'_id': True})
    trained_state_ids = list(trained_state_ids)

    for trained_state_id in trained_state_ids:
        s_trained_state = database_client.trained_state_collection.find_one({'_id': trained_state_id['_id']})
        trained_state = database_client.deserialize_entity(s_trained_state)
        system.set_trained_state(trained_state)

        test_vision_system(system, database_client, config)
