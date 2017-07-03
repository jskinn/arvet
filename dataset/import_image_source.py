import copy
import core.image_collection
import core.image_entity
import core.sequence_type
import util.database_helpers as db_help


def import_dataset_from_image_source(db_client, image_source, filter_function=None):
    """
    Read an image source, and save it in the database as an image collection.
    This is used to both save datasets from simulation,
    and to sample existing datasets into new collections.

    :param db_client: The connection to the database
    :param image_source: The image source to save
    :param filter_function: A function used to filter the images that will be part of the new collection.
    :return:
    """
    image_ids = []
    image_source.begin()
    while not image_source.is_complete():
        image, _ = image_source.get_next_image()
        if not callable(filter_function) or filter_function(image):
            if hasattr(image, 'identifier') and image.identifier is not None:
                # Image is already in the database, just store it's id
                image_ids.append(image.identifier)
            else:
                # Image is not in the database, convert it to an entity and store it.
                image_entity = core.image_entity.image_to_entity(image)
                query = db_help.query_to_dot_notation(image_entity.serialize())
                existing = db_client.image_collection.find_one(query, {'_id': True})
                if existing is None:
                    image_entity.save_image_data(db_client)
                    # Need to serialize again so we can store the newly created data ids.
                    image_ids.append(db_client.image_collection.insert(image_entity.serialize()))
                else:
                    # An identical image already exists, use that.
                    image_ids.append(existing['_id'])
    if len(image_ids) > 0:
        s_collection = core.image_collection.ImageCollection.create_serialized(
            image_ids=image_ids,
            sequence_type=image_source.sequence_type
        )
        query = db_help.query_to_dot_notation(copy.deepcopy(s_collection))
        if db_client.image_source_collection.find_one(query, {'_id': True}) is None:
            db_client.image_source_collection.insert(s_collection)
