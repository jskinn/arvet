#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
import traceback
import numpy as np
import pymongo.collection
import cv2
import config.global_configuration as global_conf
import database.client
import database.entity_registry


def remove_orphan_images(db_client: database.client.DatabaseClient, dry_run=False):
    """
    Remove all the images that don't appear in an image collection
    :param db_client: The database client
    :param dry_run: Don't remove anything, just print the number of images to remove
    :return:
    """
    # Collect all the image ids that do appear in a image collection
    image_ids = set()
    image_sources = db_client.image_source_collection.find({'images': {'$exists': True}}, {'images': True})
    for s_image_collection in image_sources:
        image_ids |= {img_id for _, img_id in s_image_collection['images']}
    # Remove the images that don't appear in the above list.
    count = db_client.image_collection.find({'_id': {'$nin': list(image_ids)}}, {'_id': True}).count()
    logging.getLogger(__name__).info("Removing {0} images".format(count))
    if not dry_run:
        db_client.image_collection.remove({'_id': {'$nin': list(image_ids)}})


def recalculate_derivative_metadata(db_client: database.client.DatabaseClient):
    """
    Update the images in the database to
    :return:
    """
    all_images = db_client.image_collection.find({'label_data'})
    for s_image in all_images:
        image = db_client.deserialize_entity(s_image)
        if image.labels_data is not None:
            # Recalculate label bounding boxes
            for idx, labelled_object in enumerate(image.metadata.labelled_objects):
                color = labelled_object.label_color
                label_points = cv2.findNonZero(np.asarray(np.all(image.labels_data == color, axis=2), dtype='uint8'))
                db_client.image_collection.update({'_id': image.identifier}, {
                    '$set': {
                        'metadata.labelled_objects.{}.bounding_box'.format(idx): cv2.boundingRect(label_points)
                    }
                })


def check_collection(collection: pymongo.collection.Collection, db_client: database.client.DatabaseClient):
    """
    Check all the entities in a collection
    :param collection:
    :param db_client:
    :return:
    """
    all_entities = collection.find()
    for s_entity in all_entities:
        # patch the entity type if appropriate
        if '.' not in s_entity['_type']:
            qual_types = database.entity_registry.find_potential_entity_classes(s_entity['_type'])
            if len(qual_types) == 1 and qual_types[0] != s_entity['_type']:
                logging.getLogger(__name__).error("Entity {0} had unqualified type {1}".format(
                    s_entity['_id'], s_entity['_type']))
                collection.update_one({'_id': s_entity['_id']}, {'$set': {'_type': qual_types[0]}})

        # Try and deserialize the entity, and validate it if we succeed
        try:
            entity = db_client.deserialize_entity(s_entity)
        except Exception:
            entity = None
            logging.getLogger(__name__).error(
                "Exception occurred deserializing object {0}:\n{1}".format(s_entity['_id'], traceback.format_exc()))

        if entity is not None and hasattr(entity, 'validate'):
            if not entity.validate():
                logging.getLogger(__name__).error("Entity {0} ({1}) failed validation".format(
                    entity.identifier, s_entity['_type']))


def main(check_collections: bool = True, remove_orphans: bool = False, recalculate_metadata: bool = False):
    """
    Verify the state of the database
    :return:
    """
    config = global_conf.load_global_config('config.yml')
    if __name__ == '__main__':
        logging.config.dictConfig(config['logging'])
    db_client = database.client.DatabaseClient(config=config)

    if remove_orphans:
        remove_orphan_images(db_client)

    if recalculate_metadata:
        recalculate_derivative_metadata(db_client)

    # Patch saved entity types to fully-qualified names
    if check_collections:
        logging.getLogger(__name__).info('Checking experiments...')
        check_collection(db_client.experiments_collection, db_client)
        logging.getLogger(__name__).info('Checking trainers...')
        check_collection(db_client.trainer_collection, db_client)
        logging.getLogger(__name__).info('Checking trainees...')
        check_collection(db_client.trainee_collection, db_client)
        logging.getLogger(__name__).info('Checking systems...')
        check_collection(db_client.system_collection, db_client)
        logging.getLogger(__name__).info('Checking benchmarks...')
        check_collection(db_client.benchmarks_collection, db_client)
        logging.getLogger(__name__).info('Checking image sources...')
        check_collection(db_client.image_source_collection, db_client)
        # check_collection(db_client.image_collection, db_client)    # This is covered by image sources
        logging.getLogger(__name__).info('Checking trials...')
        check_collection(db_client.trials_collection, db_client)
        logging.getLogger(__name__).info('Checking results...')
        check_collection(db_client.results_collection, db_client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify the database. '
                    'This loads all the entites in all collections and calls their validate method. '
                    'It also optionally removes images that are not part of a collection, '
                    'and/or recalcult')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip the stage ')
    parser.add_argument('--remove_orphans', action='store_true',
                        help='Remove images from the database that do not appear in an image collection.')
    parser.add_argument('--recalculate_metadata', action='store_true',
                        help='Recalculate certain derived metadata for each image, '
                             'such as bounding boxes from per-pixel labels.')

    args = parser.parse_args()
    main(check_collections=not args.skip_validation,
         remove_orphans=args.remove_orphans,
         recalculate_metadata=args.recalculate_metadata)
