import copy
import core.image_collection
import core.image_entity
import core.sequence_type
import util.database_helpers as db_help


def import_dataset_from_image_source(db_client, image_source):
    """
    Read an image source, and save it in hte database as an image collection.
    :param db_client: The connection to the database
    :param image_source: The image source to save
    :return:
    """
    image_ids = []
    if image_source.is_stored_in_database():
        # Stop early, this image source is already in the database.
        return
    image_source.begin()
    while not image_source.is_complete():
        image = image_source.get_next_image()
        image_entity = core.image_entity.image_to_entity(image)
        s_image = image_entity.serialize()
        query = db_help.query_to_dot_notation(copy.deepcopy(s_image))
        existing = db_client.image_collection.find_one(query, {'_id': True})
        if existing is None:
            image_entity.save_image_data(db_client)
            image_ids.append(db_client.image_collection.insert(s_image))
        else:
            image_ids.append(existing['_id'])
    s_collection = core.image_collection.ImageCollection.create_serialized(
        image_ids=image_ids,
        sequence_type=image_source.sequence_type
    )
    query = db_help.query_to_dot_notation(copy.deepcopy(s_collection))
    if db_client.image_source_collection.find_one(query, {'_id': True}) is None:
        db_client.image_source_collection.insert(s_collection)
