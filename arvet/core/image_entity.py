# Copyright (c) 2017, John Skinner
import xxhash
import numpy as np
import copy
import pickle
import arvet.database.entity
import arvet.util.database_helpers as db_help
import arvet.metadata.image_metadata as imeta
import arvet.core.image


class ImageEntity(arvet.core.image.Image, arvet.database.entity.Entity):

    def __init__(self, data_id=None, depth_id=None, ground_truth_depth_id=None, labels_id=None, world_normals_id=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._data_id = data_id
        self._depth_id = depth_id
        self._gt_depth_id = ground_truth_depth_id
        self._labels_id = labels_id
        self._world_normals_id = world_normals_id

    def save_image_data(self, db_client, force_update=False):
        """
        Store the data for this image in the GridFS.
        You need to call this before serializing the image.

        :param db_client: The database client. We need this for access to GridFS.
        :param force_update: Store the data even if it already exists.
        :return: void
        """
        if force_update or self._data_id is None:
            self._data_id = db_client.grid_fs.put(pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL))
        if self.depth_data is not None and (force_update or self._depth_id is None):
            self._depth_id = db_client.grid_fs.put(pickle.dumps(self.depth_data, protocol=pickle.HIGHEST_PROTOCOL))
        if self.ground_truth_depth_data is not None and (force_update or self._gt_depth_id is None):
            self._gt_depth_id = db_client.grid_fs.put(pickle.dumps(self.ground_truth_depth_data,
                                                                   protocol=pickle.HIGHEST_PROTOCOL))
        if self.labels_data is not None and (force_update or self._labels_id is None):
            self._labels_id = db_client.grid_fs.put(pickle.dumps(self.labels_data, protocol=pickle.HIGHEST_PROTOCOL))
        if self.world_normals_data is not None and (force_update or self._world_normals_id is None):
            self._world_normals_id = db_client.grid_fs.put(pickle.dumps(self.world_normals_data,
                                                                        protocol=pickle.HIGHEST_PROTOCOL))

    def validate(self):
        if self.data is None:
            # Raw RGB data is null
            return False
        if not self.data.shape[1] == self.metadata.width or not self.data.shape[0] == self.metadata.height:
            # Image dimensions are wrong
            return False
        if not self.metadata.hash == xxhash.xxh64(self.data).digest():
            # Hash is wrong
            return False
        if self.depth_data is not None and self.metadata.average_scene_depth != np.mean(self.depth_data):
            # Average scene depth does not match actual depth data
            return False
        # TODO: Verify bounding boxes
        return True

    def serialize(self):
        serialized = super().serialize()
        serialized['data'] = self._data_id
        serialized['metadata'] = self.metadata.serialize()
        serialized['additional_metadata'] = copy.deepcopy(self.additional_metadata)
        serialized['depth_data'] = self._depth_id
        serialized['ground_truth_depth_data'] = self._gt_depth_id
        serialized['labels_data'] = self._labels_id
        serialized['world_normals_data'] = self._world_normals_id
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'metadata' in serialized_representation:
            kwargs['metadata'] = imeta.ImageMetadata.deserialize(serialized_representation['metadata'])
        if 'additional_metadata' in serialized_representation:
            kwargs['additional_metadata'] = serialized_representation['additional_metadata']

        if 'data' in serialized_representation:
            kwargs['data_id'] = serialized_representation['data']
            if kwargs['data_id'] is not None:
                kwargs['data'] = pickle.loads(db_client.grid_fs.get(kwargs['data_id']).read())
        if 'depth_data' in serialized_representation:
            kwargs['depth_id'] = serialized_representation['depth_data']
            if kwargs['depth_id'] is not None:
                kwargs['depth_data'] = pickle.loads(db_client.grid_fs.get(kwargs['depth_id']).read())
        if 'ground_truth_depth_data' in serialized_representation:
            kwargs['ground_truth_depth_id'] = serialized_representation['ground_truth_depth_data']
            if kwargs['ground_truth_depth_id'] is not None:
                kwargs['ground_truth_depth_data'] = pickle.loads(db_client.grid_fs.get(
                    kwargs['ground_truth_depth_id']).read())
        if 'labels_data' in serialized_representation:
            kwargs['labels_id'] = serialized_representation['labels_data']
            if kwargs['labels_id'] is not None:
                kwargs['labels_data'] = pickle.loads(db_client.grid_fs.get(kwargs['labels_id']).read())
        if 'world_normals_data' in serialized_representation:
            kwargs['world_normals_id'] = serialized_representation['world_normals_data']
            if kwargs['world_normals_id'] is not None:
                kwargs['world_normals_data'] = pickle.loads(db_client.grid_fs.get(kwargs['world_normals_id']).read())
        return super().deserialize(serialized_representation, db_client, **kwargs)


class StereoImageEntity(arvet.core.image.StereoImage, ImageEntity):

    def __init__(self, left_data_id=None, left_depth_id=None, left_ground_truth_depth_id=None,
                 left_labels_id=None, left_world_normals_id=None,
                 right_data_id=None, right_depth_id=None, right_ground_truth_depth_id=None,
                 right_labels_id=None, right_world_normals_id=None, **kwargs):
        # Fiddle arguments to make the left one the base
        super().__init__(data_id=left_data_id,
                         depth_id=left_depth_id,
                         ground_truth_depth_id=left_ground_truth_depth_id,
                         labels_id=left_labels_id,
                         world_normals_id=left_world_normals_id,
                         **kwargs)
        self._right_data_id = right_data_id
        self._right_depth_id = right_depth_id
        self._right_gt_depth_id = right_ground_truth_depth_id
        self._right_labels_id = right_labels_id
        self._right_world_normals_id = right_world_normals_id

    def save_image_data(self, db_client, force_update=False):
        """
        Store the data for this image in the GridFS.
        You need to call this before serializing the image.
        Overridden to save the right hand image as well

        :param db_client: The database client. We need this for access to GridFS.
        :param force_update: Save data to the database even if it already exists. Default False.
        :return: void
        """
        super().save_image_data(db_client, force_update)
        if force_update or self._right_data_id is None:
            self._right_data_id = db_client.grid_fs.put(pickle.dumps(self.right_data,
                                                                     protocol=pickle.HIGHEST_PROTOCOL))
        if self.right_depth_data is not None and (force_update or self._right_depth_id is None):
            self._right_depth_id = db_client.grid_fs.put(pickle.dumps(self.right_depth_data,
                                                                      protocol=pickle.HIGHEST_PROTOCOL))
        if self.right_ground_truth_depth_data is not None and (force_update or self._right_gt_depth_id is None):
            self._right_gt_depth_id = db_client.grid_fs.put(pickle.dumps(self.right_ground_truth_depth_data,
                                                                         protocol=pickle.HIGHEST_PROTOCOL))
        if self.right_labels_data is not None and (force_update or self._right_labels_id is None):
            self._right_labels_id = db_client.grid_fs.put(pickle.dumps(self.right_labels_data,
                                                                       protocol=pickle.HIGHEST_PROTOCOL))
        if self.right_world_normals_data is not None and (force_update or self._right_world_normals_id is None):
            self._right_world_normals_id = db_client.grid_fs.put(pickle.dumps(self.right_world_normals_data,
                                                                              protocol=pickle.HIGHEST_PROTOCOL))

    def validate(self):
        if not super().validate():
            return False
        if self.right_data is None:
            return False
        return True

    def serialize(self):
        serialized = super().serialize()

        # fiddle the serialized version for left and right images
        fiddle_keys = ['data', 'depth_data', 'ground_truth_depth_data', 'labels_data', 'world_normals_data']
        for key in fiddle_keys:
            serialized['left_' + key] = serialized[key]
            del serialized[key]

        serialized['right_data'] = self._right_data_id
        serialized['right_depth_data'] = self._right_depth_id
        serialized['right_ground_truth_depth_data'] = self._right_gt_depth_id
        serialized['right_labels_data'] = self._right_labels_id
        serialized['right_world_normals_data'] = self._right_world_normals_id
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):

        if 'left_data' in serialized_representation:
            kwargs['left_data_id'] = serialized_representation['left_data']
            if kwargs['left_data_id'] is not None:
                kwargs['left_data'] = pickle.loads(db_client.grid_fs.get(kwargs['left_data_id']).read())
        if 'left_depth_data' in serialized_representation:
            kwargs['left_depth_id'] = serialized_representation['left_depth_data']
            if kwargs['left_depth_id'] is not None:
                kwargs['left_depth_data'] = pickle.loads(db_client.grid_fs.get(kwargs['left_depth_id']).read())
        if 'left_ground_truth_depth_data' in serialized_representation:
            kwargs['left_ground_truth_depth_id'] = serialized_representation['left_ground_truth_depth_data']
            if kwargs['left_ground_truth_depth_id'] is not None:
                kwargs['left_ground_truth_depth_data'] = pickle.loads(db_client.grid_fs.get(
                    kwargs['left_ground_truth_depth_id']).read())
        if 'left_labels_data' in serialized_representation:
            kwargs['left_labels_id'] = serialized_representation['left_labels_data']
            if kwargs['left_labels_id'] is not None:
                kwargs['left_labels_data'] = pickle.loads(db_client.grid_fs.get(kwargs['left_labels_id']).read())
        if 'left_world_normals_data' in serialized_representation:
            kwargs['left_world_normals_id'] = serialized_representation['left_world_normals_data']
            if kwargs['left_world_normals_id'] is not None:
                kwargs['left_world_normals_data'] = pickle.loads(db_client.grid_fs.get(
                    kwargs['left_world_normals_id']).read())

        if 'right_data' in serialized_representation:
            kwargs['right_data_id'] = serialized_representation['right_data']
            if kwargs['right_data_id'] is not None:
                kwargs['right_data'] = pickle.loads(db_client.grid_fs.get(kwargs['right_data_id']).read())
        if 'right_depth_data' in serialized_representation:
            kwargs['right_depth_id'] = serialized_representation['right_depth_data']
            if kwargs['right_depth_id'] is not None:
                kwargs['right_depth_data'] = pickle.loads(db_client.grid_fs.get(kwargs['right_depth_id']).read())
        if 'right_ground_truth_depth_data' in serialized_representation:
            kwargs['right_ground_truth_depth_id'] = serialized_representation['right_ground_truth_depth_data']
            if kwargs['right_ground_truth_depth_id'] is not None:
                kwargs['right_ground_truth_depth_data'] = pickle.loads(db_client.grid_fs.get(
                    kwargs['right_ground_truth_depth_id']).read())
        if 'right_labels_data' in serialized_representation:
            kwargs['right_labels_id'] = serialized_representation['right_labels_data']
            if kwargs['right_labels_id'] is not None:
                kwargs['right_labels_data'] = pickle.loads(db_client.grid_fs.get(kwargs['right_labels_id']).read())
        if 'right_world_normals_data' in serialized_representation:
            kwargs['right_world_normals_id'] = serialized_representation['right_world_normals_data']
            if kwargs['right_world_normals_id'] is not None:
                kwargs['right_world_normals_data'] = pickle.loads(db_client.grid_fs.get(
                    kwargs['right_world_normals_id']).read())
        return super().deserialize(serialized_representation, db_client, **kwargs)


def image_to_entity(image):
    """
    Convert an image object to an image entity.
    Handles parameter renames and the diamond inheritance between stereo images and normal images
    to create the correct kind of image entity.
    Does nothing if the argument is already and image entity
    :param image: An image object
    :return:
    """
    if isinstance(image, ImageEntity):
        return image
    elif isinstance(image, arvet.core.image.StereoImage):
        return StereoImageEntity(left_data=image.left_data,
                                 right_data=image.right_data,
                                 metadata=image.metadata,
                                 additional_metadata=image.additional_metadata,
                                 left_depth_data=image.left_depth_data,
                                 left_ground_truth_depth_data=image.left_ground_truth_depth_data,
                                 left_labels_data=image.left_labels_data,
                                 left_world_normals_data=image.left_world_normals_data,
                                 right_depth_data=image.right_depth_data,
                                 right_ground_truth_depth_data=image.right_ground_truth_depth_data,
                                 right_labels_data=image.right_labels_data,
                                 right_world_normals_data=image.right_world_normals_data)
    elif isinstance(image, arvet.core.image.Image):
        return ImageEntity(data=image.data,
                           metadata=image.metadata,
                           additional_metadata=image.additional_metadata,
                           depth_data=image.depth_data,
                           ground_truth_depth_data=image.ground_truth_depth_data,
                           labels_data=image.labels_data,
                           world_normals_data=image.world_normals_data)
    else:
        return image


def save_image(db_client, image):
    """
    Save an image to the database.
    First checks if the image already exists,
    and does not insert if it does.
    :param db_client: A database client object to use to save the image.
    :param image: An image entity or image object to be saved to the database
    :return: the id of the image in the database
    """
    if not isinstance(image, ImageEntity):
        if isinstance(image, arvet.core.image.Image):
            image = image_to_entity(image)
        else:
            return None
    existing_query = image.serialize()

    # Don't look at the GridFS links when determining if the image exists, only use metadata.
    delete_keys = ['data', 'depth_data', 'ground_truth_depth_data', 'labels_data', 'world_normals_data']
    for key in delete_keys:
        if key in existing_query:
            del existing_query[key]
        if 'left_' + key in existing_query:
            del existing_query['left_' + key]
        if 'right_' + key in existing_query:
            del existing_query['right_' + key]
    db_help.query_to_dot_notation(existing_query, flatten_arrays=True)

    existing = db_client.image_collection.find_one(existing_query, {'_id': True})
    if existing is None:
        image.save_image_data(db_client)
        # Need to serialize again so we can store the newly created data ids.
        return db_client.image_collection.insert_one(image.serialize()).inserted_id
    else:
        # An identical image already exists, use that.
        return existing['_id']


def delete_image(db_client, image_id):
    """
    Delete an image by id, including removing the data stored in GridFS.
    :param db_client: The database client
    :param image_id: The image id.
    :return:
    """
    # Generate the keys that may point to GridFS data
    delete_keys = ['data', 'depth_data', 'ground_truth_depth_data', 'labels_data', 'world_normals_data']
    delete_keys = [prefix + key for key in delete_keys for prefix in ('', 'left_', '_right_')]

    s_image = db_client.image_collection.find_one({'_id': image_id}, {key: True for key in delete_keys})

    # Delete the data from the GridFS keys
    for key in delete_keys:
        if key in s_image:
            db_client.grid_fs.delete(s_image[key])
        if 'left_' + key in s_image:
            db_client.grid_fs.delete(s_image['left_' + key])
        if 'right_' + key in s_image:
            db_client.grid_fs.delete(s_image['right_' + key])

    db_client.image_collection.delete_one({'_id': image_id})
