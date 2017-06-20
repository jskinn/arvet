import os
import os.path
import copy
import pickle
import database.entity
import util.transform as tf
import metadata.image_metadata as imeta
import core.image


class ImageEntity(core.image.Image, database.entity.Entity):

    def __init__(self, data_id=None, depth_id=None, labels_id=None, world_normals_id=None, **kwargs):
        super().__init__(**kwargs)
        self._data_id = data_id
        self._depth_id = depth_id
        self._labels_id = labels_id
        self._world_normals_id = world_normals_id

    def save_image_data(self, db_client, force_update=False):
        """
        Store the data for this image in the GridFS.
        You need to call this before serializing the image.

        :param db_client: The database client. We need this for access to GridFS.
        :return: void
        """
        if force_update or self._data_id is None:
            self._data_id = db_client.grid_fs.put(pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL))
        if self.depth_data is not None and (force_update or self._depth_id is None):
            self._depth_id = db_client.grid_fs.put(pickle.dumps(self.depth_data, protocol=pickle.HIGHEST_PROTOCOL))
        if self.labels_data is not None and (force_update or self._labels_id is None):
            self._labels_id = db_client.grid_fs.put(pickle.dumps(self.labels_data, protocol=pickle.HIGHEST_PROTOCOL))
        if self.world_normals_data is not None and (force_update or self._world_normals_id is None):
            self._world_normals_id = db_client.grid_fs.put(pickle.dumps(self.world_normals_data,
                                                                        protocol=pickle.HIGHEST_PROTOCOL))

    def validate(self):
        if not os.path.isfile(self.data):
            return False
        if self.depth_data is not None and not os.path.isfile(self.depth_data):
            return False
        if self.labels_data is not None and not os.path.isfile(self.labels_data):
            return False
        if self.depth_data is not None and not os.path.isfile(self.depth_data):
            return False
        return True

    def serialize(self):
        serialized = super().serialize()
        serialized['data'] = self._data_id
        serialized['camera_pose'] = self.camera_pose.serialize()
        serialized['metadata'] = self.metadata.serialize()
        serialized['additional_metadata'] = copy.deepcopy(self.additional_metadata)
        serialized['depth_data'] = self._depth_id
        serialized['labels_data'] = self._labels_id
        serialized['world_normals_data'] = self._world_normals_id
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'camera_pose' in serialized_representation:
            kwargs['camera_pose'] = tf.Transform.deserialize(serialized_representation['camera_pose'])
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
        if 'labels_data' in serialized_representation:
            kwargs['labels_id'] = serialized_representation['labels_data']
            if kwargs['labels_id'] is not None:
                kwargs['labels_data'] = pickle.loads(db_client.grid_fs.get(kwargs['labels_id']).read())
        if 'world_normals_data' in serialized_representation:
            kwargs['world_normals_id'] = serialized_representation['world_normals_data']
            if kwargs['world_normals_id'] is not None:
                kwargs['world_normals_data'] = pickle.loads(db_client.grid_fs.get(kwargs['world_normals_id']).read())
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def from_image(cls, image):
        return cls(data=image.data,
                   camera_location=image.camera_location,
                   camera_orientation=image.camera_orientation,
                   additional_metadata=image.additional_metadata,
                   depth_data=image.depth_data,
                   labels_data=image.labels_data,
                   world_normals_data=image.world_normals_data)


class StereoImageEntity(core.image.StereoImage, ImageEntity):

    def __init__(self, left_data_id=None, left_depth_id=None, left_labels_id=None, left_world_normals_id=None,
                 right_data_id=None, right_depth_id=None, right_labels_id=None, right_world_normals_id=None, **kwargs):
        # Fiddle arguments ot make the left one the base
        super().__init__(data_id=left_data_id,
                         depth_id=left_depth_id,
                         labels_id=left_labels_id,
                         world_normals_id=left_world_normals_id,
                         **kwargs)
        self._right_data_id = right_data_id
        self._right_depth_id = right_depth_id
        self._right_labels_id = right_labels_id
        self._right_world_normals_id = right_world_normals_id

    def save_image_data(self, db_client, force_update=False):
        """
        Store the data for this image in the GridFS.
        You need to call this before serializing the image.
        Overridden to save the right hand image as well

        :param db_client: The database client. We need this for access to GridFS.
        :return: void
        """
        super().save_image_data(db_client, force_update)
        if force_update or self._right_data_id is None:
            self._right_data_id = db_client.grid_fs.put(pickle.dumps(self.right_data,
                                                                     protocol=pickle.HIGHEST_PROTOCOL))
        if self.right_depth_data is not None and (force_update or self._right_depth_id is None):
            self._right_depth_id = db_client.grid_fs.put(pickle.dumps(self.right_depth_data,
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
        if not os.path.isfile(self.right_data):
            return False
        if self.depth_data is not None and not os.path.isfile(self.right_depth_data):
            return False
        if self.labels_data is not None and not os.path.isfile(self.right_labels_data):
            return False
        if self.depth_data is not None and not os.path.isfile(self.right_depth_data):
            return False
        return True

    def serialize(self):
        serialized = super().serialize()

        # fiddle the serialized version for left and right images
        fiddle_keys = ['data', 'camera_pose', 'depth_data', 'labels_data', 'world_normals_data']
        for key in fiddle_keys:
            serialized['left_' + key] = serialized[key]
            del serialized[key]

        serialized['right_data'] = self._right_data_id
        serialized['right_camera_pose'] = self.right_camera_pose.serialize()
        serialized['right_depth_data'] = self._right_depth_id
        serialized['right_labels_data'] = self._right_labels_id
        serialized['right_world_normals_data'] = self._right_world_normals_id
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):

        if 'left_data' in serialized_representation:
            kwargs['left_data_id'] = serialized_representation['left_data']
            if kwargs['left_data_id'] is not None:
                kwargs['left_data'] = pickle.loads(db_client.grid_fs.get(kwargs['left_data_id']).read())
        if 'left_camera_pose' in serialized_representation:
            kwargs['left_camera_pose'] = tf.Transform.deserialize(serialized_representation['left_camera_pose'])
        if 'left_depth_data' in serialized_representation:
            kwargs['left_depth_id'] = serialized_representation['left_depth_data']
            if kwargs['left_depth_id'] is not None:
                kwargs['left_depth_data'] = pickle.loads(db_client.grid_fs.get(kwargs['left_depth_id']).read())
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
        if 'right_camera_pose' in serialized_representation:
            kwargs['right_camera_pose'] = tf.Transform.deserialize(serialized_representation['right_camera_pose'])
        if 'right_depth_data' in serialized_representation:
            kwargs['right_depth_id'] = serialized_representation['right_depth_data']
            if kwargs['right_depth_id'] is not None:
                kwargs['right_depth_data'] = pickle.loads(db_client.grid_fs.get(kwargs['right_depth_id']).read())
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
