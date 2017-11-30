# Copyright (c) 2017, John Skinner
import copy
import argus.util.dict_utils as du
import argus.util.database_helpers as dh


def train_system(trainer, db_client, consolidated_folder, settings_variations, dataset_variations):
    """
    Perform training on a particular trainable vision system
    :param trainer:
    :param db_client:
    :param consolidated_folder:
    :param settings_variations:
    :param dataset_variations:
    :return:
    """
    for system_config in settings_variations:
        trainer.set_settings(system_config)

        for dataset_criteria in dataset_variations:
            dataset_ids = db_client.dataset_collection.find(
                du.defaults(trainer.get_training_dataset_criteria(), dataset_criteria), {'_id': True})
            dataset_ids = [temp['_id'] for temp in dataset_ids]

            # Make sure this trained state doesn't already exist.
            existing_query = dh.query_to_dot_notation({'settings': copy.deepcopy(system_config)})
            existing_query['datasets'] = {'$all': dataset_ids}
            existing_count = db_client.trained_state_collection.find(existing_query).count()

            if existing_count <= 0:
                for dataset_id in dataset_ids:
                    s_dataset = db_client.dataset_collection.find_one({'_id': dataset_id})
                    dataset = db_client.deserialize_entity(s_dataset)
                    dataset_images = dataset.load_images(db_client)
                    trainer.add_dataset(dataset_images)

                trained_state = trainer.train_system()

                # Add the trained state to the database, then consolidate files.
                id_ = db_client.trained_state_collection.insert(trained_state.serialize())
                trained_state.refresh_id(id_)
                trained_state.consolidate_files(consolidated_folder)

                # store the updated filenames in the database,
                update_query = dh.query_to_dot_notation(trained_state.serialize())
                del update_query['_id']     # Cannot update the id
                update_query = {'$set': update_query}
                db_client.trained_state_collection.update({'_id': id_}, update_query)
