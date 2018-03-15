# Copyright (c) 2017, John Skinner
import abc
import os
import numpy as np
import logging
import pickle
import arvet.database.entity
import arvet.database.entity_registry as entity_registry
import arvet.core.image
import arvet.core.sequence_type
import arvet.core.image_source
import arvet.core.image_entity
import arvet.util.database_helpers as dh


class ImageCollection(arvet.core.image_source.ImageSource, arvet.database.entity.Entity, metaclass=abc.ABCMeta):
    """
    A collection of images stored in the database.
    This can be a sequential set of images like a video, or a random sampling of different pictures.
    """

    def __init__(self, images, type_, db_client_, id_=None, **kwargs):
        super().__init__(id_=id_, **kwargs)

        self._images = images
        if (isinstance(type_, arvet.core.sequence_type.ImageSequenceType) and
                type_ is not arvet.core.sequence_type.ImageSequenceType.INTERACTIVE):
            # image collections cannot be interactive
            self._sequence_type = type_
        else:
            self._sequence_type = arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL
        self._timestamps = sorted(self._images.keys())
        self._current_index = 0

        self._db_client = db_client_
        if len(images) > 0:
            image_ids = list(images.values())
            self._is_depth_available = (db_client_.image_collection.find({
                '_id': {'$in': image_ids},
                'depth_data': None,
                'left_depth_data': None,
            }).count() <= 0)
            self._is_labels_image_available = (db_client_.image_collection.find({
                '_id': {'$in': image_ids},
                'labels_data': None,
                'left_labels_data': None
            }).count() <= 0)
            self._is_bboxes_available = (db_client_.image_collection.find({
                '_id': {'$in': image_ids},
                'metadata.labelled_objects': []
            }).count() <= 0)
            self._is_normals_available = (db_client_.image_collection.find({
                '_id': {'$in': image_ids},
                'world_normals_data': None,
                'left_world_normals_data': None
            }).count() <= 0)
            self._is_stereo_available = (db_client_.image_collection.find({
                '_id': {'$in': image_ids},
                'left_data': None,
                'right_data': None
            }).count() <= 0)
        else:
            self._is_depth_available = False
            self._is_labels_image_available = False
            self._is_bboxes_available = False
            self._is_normals_available = False
            self._is_stereo_available = False

        # Get some metadata from the first image in the collection
        s_first_image = db_client_.image_collection.find_one({'_id': images[self._timestamps[0]]})
        first_image = db_client_.deserialize_entity(s_first_image)
        self._camera_intrinsics = first_image.metadata.camera_intrinsics
        self._stereo_baseline = None
        if first_image.metadata.right_camera_pose is not None:
            self._stereo_baseline = np.linalg.norm(first_image.metadata.camera_pose.location -
                                                   first_image.metadata.right_camera_pose.location)

    def __len__(self):
        """
        The length of the image collection
        :return:
        """
        return len(self._images)

    def __iter__(self):
        """
        Iterator for the image collection.
        Returns the iterator over the inner images dict
        :return:
        """
        return self._images.items()

    def __getitem__(self, item):
        """
        Allow index-based access. Why not.
        This is the same as get
        :param item:
        :return:
        """
        return self.get(item)

    @property
    def sequence_type(self):
        """
        Get the type of image sequence produced by this image source.
        This is determined when creating the image collection
        It is useful for determining which sources can run with which algorithms.
        Image collections can be NON_SEQUENTIAL or SEQUENTIAL, but not INTERACTIVE
        :return: The image sequence type enum
        :rtype arvet.core.image_sequence.ImageSequenceType:
        """
        return self._sequence_type

    @property
    def timestamps(self):
        """
        Get the list of timestamps/indexes in this collection, in order.
        They are the list of valid keys to get and __getitem__,
        all others return None
        :return:
        """
        return self._timestamps

    def begin(self):
        """
        Start producing images.
        Resets the current index to the start
        :return: True
        """
        self._current_index = 0
        return True

    def get(self, index):
        """
        A getter for random access, since we're storing a list
        :param index:
        :return:
        """
        if index in self._images:
            # Check if the image is cached on the file system
            cache_filename = get_cache_filename(self._db_client.temp_folder, self._images[index])
            if os.path.isfile(cache_filename):
                with open(cache_filename, 'rb') as imfile:
                    return pickle.load(imfile)
            # Go to the database if we haven't got a local file
            return dh.load_object(self._db_client, self._db_client.image_collection, self._images[index])
        return None

    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images

        :return: An Image object (see arvet.core.image) or None, and a timestamp or None
        """
        if not self.is_complete():
            timestamp = self._timestamps[self._current_index]
            image = self.get(timestamp)
            self._current_index += 1
            return image, timestamp
        return None, None

    def is_complete(self):
        """
        Have we got all the images from this source?
        Some sources are infinite, some are not,
        and this method lets those that are not end the iteration.
        :return: True if there are more images to get, false otherwise.
        """
        return self._current_index >= len(self._timestamps)

    @property
    def supports_random_access(self):
        """
        Image collections support random access, they are a list of images
        :return:
        """
        return True

    @property
    def is_depth_available(self):
        """
        Do the images in this sequence include depth
        :return: True if depth is available for all images in this sequence
        """
        return self._is_depth_available

    @property
    def is_per_pixel_labels_available(self):
        """
        Do images from this image source include object lables
        :return: True if this image source can produce object labels for each image
        """
        return self._is_labels_image_available

    @property
    def is_labels_available(self):
        """
        Do images from this source include object bounding boxes in their metadata.
        :return: True iff the image metadata includes bounding boxes
        """
        return self._is_bboxes_available

    @property
    def is_normals_available(self):
        """
        Do images from this image source include world normals
        :return: True if images have world normals associated with them 
        """
        return self._is_normals_available

    @property
    def is_stereo_available(self):
        """
        Can this image source produce stereo images.
        Some algorithms only run with stereo images
        :return:
        """
        return self._is_stereo_available

    @property
    def is_stored_in_database(self):
        """
        Do this images from this source come from the database.
        Image collections are always stored in the database
        :return:
        """
        return True

    def get_camera_intrinsics(self):
        """
        Get the camera intrinisics for this image collection.
        At the moment it assumes it is the same for all images,
        and just reads it from the first.
        When I have effective metadata aggregation, read it from that.
        :return:
        """
        return self._camera_intrinsics

    def get_stereo_baseline(self):
        """
        Get the distance between the stereo cameras, or None if the images in this collection are not stereo.
        :return:
        """
        return self._stereo_baseline

    def warmup_cache(self):
        """
        Create local files to store all the images for faster loading,
        and reuse across multiple run-tasks
        :return:
        """
        batch_size = 1000   # Number of images to request at once
        timestamps = sorted(self._images.keys())
        os.makedirs(os.path.join(self._db_client.temp_folder, 'image_cache'), exist_ok=True)
        for idx in range(len(timestamps) // batch_size + 1):
            timestamps_to_load = timestamps[idx * batch_size:min(len(timestamps), (idx + 1) * batch_size)]
            images_to_load = [self._images[stamp] for stamp in timestamps_to_load]

            images_cursor = self._db_client.image_collection.find({'_id': {'$in': images_to_load}})
            for s_image in images_cursor:
                image = self._db_client.deserialize_entity(s_image)
                cache_filename = get_cache_filename(self._db_client.temp_folder, image.identifier)
                if not os.path.isfile(cache_filename):
                    with open(cache_filename, 'wb') as imfile:
                        pickle.dump(image, imfile, protocol=pickle.HIGHEST_PROTOCOL)

    def validate(self):
        """
        The image sequence is valid iff all the contained images are valid
        Only count the images that have a validate method
        :return: True if all the images are valid, false if not
        """
        with self:
            while not self.is_complete():
                image, _ = self.get_next_image()
                if hasattr(image, 'validate'):
                    if not image.validate():
                        return False
        return True

    def serialize(self):
        serialized = super().serialize()
        # Only include the image IDs here, they'll get turned back into objects for us
        serialized['images'] = [[stamp, image_id] for stamp, image_id in self._images.items()]
        if self.sequence_type is arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL:
            serialized['sequence_type'] = 'SEQ'
        else:
            serialized['sequence_type'] = 'NON'
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs) -> 'ImageCollection':
        """
        Load any collection of images.
        This handles the weird chicken-and-egg problem of deserializing
        the image collection and the individual images.

        :param serialized_representation: 
        :param db_client: An instance of arvet.database.client, from which to load the image collection
        :param kwargs: Additional arguments passed to the entity constructor.
        These will be overridden by values in serialized representation
        :return: A deserialized 
        """
        if 'images' in serialized_representation:
            kwargs['images'] = {stamp: img_id for stamp, img_id in serialized_representation['images']}
        if 'sequence_type' in serialized_representation and serialized_representation['sequence_type'] == 'SEQ':
            kwargs['type_'] = arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL
        else:
            kwargs['type_'] = arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL
        kwargs['db_client_'] = db_client
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def create_and_save(cls, db_client, image_map, sequence_type):
        """
        Make an already serialized image collection.
        Since, sometimes we have the image ids, but we don't want to have to load the objects to make the collection.
        WARNING: This can create invalid serialized image collections, since it can't check the validity of the ids.

        :param db_client: The database client, used to check image ids and for saving
        :param image_map: A map of timestamp to bson.objectid.ObjectId that refer to image objects in the database
        :param sequence_type: arvet.core.sequence_type.ImageSequenceType
        :return: The id of the newly created image collection, or None if there is an error
        """
        found_images = db_client.image_collection.find({
            '_id': {'$in': list(image_map.values())}
        }, {'_id': True}).count()
        if not found_images == len(image_map):
            logging.getLogger(__name__).warning(
                "Tried to create image collection with {0} missing ids".format(len(image_map) - found_images))
            return None
        s_images_list = [[stamp, image_id] for stamp, image_id in image_map.items()]
        s_seq_type = 'SEQ' if sequence_type is arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL else 'NON'
        existing = db_client.image_source_collection.find_one({
            '_type': entity_registry.get_type_name(cls),
            'images': {'$all': s_images_list},
            'sequence_type': s_seq_type
        }, {'_id': True})
        if existing is not None:
            return existing['_id']
        else:
            return db_client.image_source_collection.insert_one({
                '_type': entity_registry.get_type_name(cls),
                'images': s_images_list,
                'sequence_type': s_seq_type
            }).inserted_id


def delete_image_collection(db_client, image_collection_id):
    """
    A helper to delete image collections and all the images contained therein.
    Images contained in more than one collection will be retained.

    :param db_client:
    :param image_collection_id:
    :return:
    """
    # Find all the images.
    s_collection = db_client.image_source_collection.find_one({'_id': image_collection_id}, {'images': True})
    for _, image_id in s_collection['images']:
        # Find the number of image collections containing this image id
        if db_client.image_source_collection.find({'images': image_id}, limit=2).count() <= 1:
            arvet.core.image_entity.delete_image(db_client, image_id)
    db_client.image_source_collection.delete_one({'_id': image_collection_id})


def get_cache_filename(temp_dir, image_id):
    """
    Get the expected filename for an image if it has been locally cached on the filesystem.
    This can speed up loading and reduce database hits
    :param temp_dir: The temp dir, get it from the database client
    :param image_id: The image id, as a bson object
    :return:
    """
    return os.path.join(temp_dir, 'image_cache', '{0}.pickle'.format(image_id))
