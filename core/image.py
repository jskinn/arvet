import os
import os.path
import database.entity
import util.geometry as geom


#TODO: This is what Image becomes, the functionality below moved to monocular_image and stereo_image
# class Image(database.entity.Entity, metaclass=abc.ABCMeta):
#     def __init__(self, id_=None, **kwargs):
#         super().__init__(id_=id_)
#
#     @property
#     @abc.abstractmethod
#     def dataset(self):
#         """
#         Get the dataset to which the image belongs
#         :return:
#         """
#         pass
#
#     @property
#     @abc.abstractmethod
#     def filename(self):
#         """
#         Get the filename for this image.
#         :return:
#         """
#         pass
#
#     @property
#     @abc.abstractmethod
#     def metadata_filename(self):
#         """
#         Get the metadata filename for this image
#         :return:
#         """
#         pass
#
#     @property
#     @abc.abstractproperty
#     def depth_filename(self):
#         """
#         Get the filename for the scene depth image for this image.
#         Return None if no depth data is available.
#         :return:
#         """
#         pass
#
#     @property
#     @abc.abstractmethod
#     def index(self):
#         """
#         Get the index for this image
#         :return:
#         """
#         pass
#
#     @property
#     @abc.abstractmethod
#     def timestamp(self):
#         """
#         Get the
#         :return:
#         """
#         pass
#
#     @property
#     @abc.abstractmethod
#     def camera_location(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def camera_orientation(self):
#         pass
#
#     @abc.abstractmethod
#     def consolidate_files(self, new_folder):
#         pass


class Image(database.entity.Entity):
    def __init__(self, dataset, filename, index, timestamp, camera_location, camera_orientation, right_filename=None,
                 depth_filename=None, right_depth_filename=None, id_=None, **kwargs):
        super().__init__(id_=id_)
        self._dataset_id = dataset
        self._filename = filename
        self._index = index
        self._timestamp = timestamp
        self._camera_location = camera_location
        self._camera_orientation = camera_orientation
        self._right_filename = right_filename
        self._depth_filename = depth_filename
        self._right_depth_filename = right_depth_filename

    @property
    def dataset(self):
        return self._dataset_id

    @property
    def filename(self):
        return self._filename

    @property
    def metadata_filename(self):
        return self.filename + ".metadata.json"

    @property
    def index(self):
        return self._index

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def camera_location(self):
        return self._camera_location

    @property
    def camera_orientation(self):
        return self._camera_orientation

    @property
    def is_stereo(self):
        return self._right_filename is not None

    @property
    def left_filename(self):
        return self.filename

    @property
    def right_filename(self):
        return self._right_filename

    @property
    def has_depth(self):
        return self._depth_filename is not None and (not self.is_stereo or self._right_depth_filename is not None)

    @property
    def depth_filename(self):
        return self._depth_filename

    @property
    def left_depth_filename(self):
        return self.depth_filename

    @property
    def right_depth_filename(self):
        return self._right_depth_filename

    def consolidate_files(self, new_folder):
        # TODO: Be able to move into a HDF 5 file, this just changes the folder.
        _, ext = os.path.splitext(self.filename)
        new_filename = os.path.join(new_folder, self.identifier + ext)
        os.rename(self.filename, new_filename)
        self._filename = new_filename
        return {'$set': {'filename': self.filename}}

    def serialize(self):
        serialized = super().serialize()
        serialized['dataset'] = self.dataset
        serialized['filename'] = self.filename
        serialized['index'] = self.index
        serialized['timestamp'] = self.timestamp
        serialized['camera_location'] = geom.numpy_vector_to_dict(self.camera_location)
        serialized['camera_orientation'] = geom.numpy_quarternion_to_dict(self.camera_orientation)
        serialized['right_filename'] = self.right_filename
        serialized['depth_filename'] = self.depth_filename
        serialized['right_depth_filename'] = self.right_depth_filename
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'dataset' in serialized_representation:
            kwargs['dataset'] = serialized_representation['dataset']
        if 'filename' in serialized_representation:
            kwargs['filename'] = serialized_representation['filename']
        if 'index' in serialized_representation:
            kwargs['index'] = serialized_representation['index']
        if 'timestamp' in serialized_representation:
            kwargs['timestamp'] = serialized_representation['timestamp']
        if 'camera_location' in serialized_representation:
            kwargs['camera_location'] = geom.dict_vector_to_np_array(serialized_representation['camera_location'])
        if 'camera_orientation' in serialized_representation:
            kwargs['camera_orientation'] = geom.dict_quaternion_to_np_array(serialized_representation['camera_orientation'])
        if 'right_filename' in serialized_representation:
            kwargs['right_filename'] = serialized_representation['right_filename']
        if 'depth_filename' in serialized_representation:
            kwargs['depth_filename'] = serialized_representation['depth_filename']
        if 'right_depth_filename' in serialized_representation:
            kwargs['right_depth_filename'] = serialized_representation['right_depth_filename']
        return super().deserialize(serialized_representation, **kwargs)
