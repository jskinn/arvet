# Copyright (c) 2017, John Skinner
import arvet.core.image_collection
import arvet.core.image_entity
import arvet.core.sequence_type


class ImageCollectionBuilder:
    """
    A builder to create image collections within the database
    """

    def __init__(self, db_client):
        self._db_client = db_client
        self._image_ids = {}
        self._max_timestamp = None
        self._sequence_type = arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL

    def __len__(self):
        """
        Get the number of images in the builder so far
        :return:
        """
        return len(self._image_ids)

    def set_non_sequential(self):
        """
        Change the sequence type of the built sequence to non-sequential.
        The default is sequential
        :return: void
        """
        self._sequence_type = arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

    def add_image(self, image, timestamp=None):
        """
        Add an image to the growing image collection.
        Does not affect the sequence type, you may need to set that manually.
        :param image: An image_entity or image object.
        :param timestamp: The timestamp of the new image, default 1 + the previous timestamp
        :return: void
        """
        if timestamp is None:
            timestamp = self._max_timestamp + 1 if self._max_timestamp is not None else 0
        if self._max_timestamp is None or timestamp > self._max_timestamp:
            self._max_timestamp = timestamp
        if hasattr(image, 'identifier') and image.identifier is not None:
            # Image is already in the database, just store it's id
            self._image_ids[timestamp] = image.identifier
        else:
            image_id = arvet.core.image_entity.save_image(self._db_client, image)
            if image_id is not None:
                self._image_ids[timestamp] = image_id

    def add_from_image_source(self, image_source, filter_function=None, offset=0):
        """
        Read an image source, and save it in the database as an image collection.
        This is used to both save datasets from simulation,
        and to sample existing datasets into new collections.

        :param image_source: The image source to save
        :param filter_function: A function used to filter the images that will be part of the new collection.
        :param offset: An offset added to the timestamps, this allows adding from multiple image sources without
        the timestamps colliding
        :return:
        """
        if (len(self._image_ids) > 0 or
                image_source.sequence_type == arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL):
            self._sequence_type = arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL
        with image_source:
            while not image_source.is_complete():
                image, timestamp = image_source.get_next_image()
                if timestamp is not None:
                    timestamp += offset
                if not callable(filter_function) or filter_function(image):
                    self.add_image(image, timestamp)

    def save(self):
        """
        Store the image collection in the database.
        Checks if such an image collection already exists.
        :return: The id of the image collection in the database
        """
        if len(self._image_ids) > 0:
            return arvet.core.image_collection.ImageCollection.create_and_save(
                db_client=self._db_client,
                image_map=self._image_ids,
                sequence_type=self._sequence_type
            )
        return None
