#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import logging
import logging.config
import argparse
from pymodm.context_managers import no_auto_dereference

from arvet.config.global_configuration import load_global_config
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.core.image_collection import ImageCollection
from arvet.core.image import Image


def remove_orphan_images(dry_run=False):
    """
    Remove all the images that don't appear in an image collection
    :param dry_run: Don't remove anything, just print the number of images to remove
    :return:
    """
    # Collect all the image ids that do appear in a image collection
    image_ids = set()
    for image_collection in ImageCollection.objects.all():
        with no_auto_dereference(ImageCollection):
            image_ids.update(image_collection.images)

    # Remove the images that don't appear in the above list.
    queryset = Image.objects.raw({'_id': {'$nin': list(image_ids)}})
    if dry_run:
        logging.getLogger(__name__).info("Dry run would remove {0} images".format(queryset.count()))
    else:
        removed = queryset.delete()
        logging.getLogger(__name__).info("Removed {0} images".format(removed))


def main(remove_orphans: bool = False):
    """
    Verify the state of the database
    :return:
    """
    # Load the configuration
    config = load_global_config('config.yml')
    if __name__ == '__main__':
        # Only configure the logging if this is the main function, don't reconfigure
        logging.config.dictConfig(config['logging'])

    # Configure the database and the image manager
    dbconn.configure(config['database'])
    im_manager.configure(config['image_manager'])

    if remove_orphans:
        remove_orphan_images()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify the database. '
                    'This loads all the entites in all collections and calls their validate method. '
                    'It also optionally removes images that are not part of a collection, '
                    'and/or recalcult')
    parser.add_argument('--remove_orphans', action='store_true',
                        help='Remove images from the database that do not appear in an image collection.')

    args = parser.parse_args()
    main(remove_orphans=args.remove_orphans)
