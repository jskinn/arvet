import os
import os.path
import copy
import database.entity
import util.geometry as geom
import util.transform as tf
import core.image


class ImageEntity(core.image.Image, database.entity.Entity):

    def consolidate_files(self, new_folder):
        """
        Move the files associated with this image
        #TODO: The purpose of this function was to consolidate imagges from my local pc to the HPC.
        Since there are problems with using the os functions over samba shares, we need a better approach.

        :param new_folder:
        :return:
        """
        pass

    def validate(self):
        if not os.path.isfile(self.filename):
            return False
        if self.depth_filename is not None and not os.path.isfile(self.depth_filename):
            return False
        if self.labels_filename is not None and not os.path.isfile(self.labels_filename):
            return False
        if self.depth_filename is not None and not os.path.isfile(self.depth_filename):
            return False
        return True

    def serialize(self):
        serialized = super().serialize()
        serialized['filename'] = self.filename
        serialized['timestamp'] = self.timestamp
        serialized['camera_pose'] = {
            'location': geom.numpy_vector_to_dict(self.camera_location),
            'rotation': geom.numpy_quarternion_to_dict(self.camera_orientation)
        }
        serialized['additional_metadata'] = copy.deepcopy(self.additional_metadata)
        serialized['depth_filename'] = self.depth_filename
        serialized['labels_filename'] = self.labels_filename
        serialized['world_normals_filename'] = self.world_normals_filename
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'filename' in serialized_representation:
            kwargs['filename'] = serialized_representation['filename']
        if 'timestamp' in serialized_representation:
            kwargs['timestamp'] = serialized_representation['timestamp']
        if 'camera_pose' in serialized_representation:
            loc = None
            rot = None
            if 'location' in serialized_representation['camera_pose']:
                loc = geom.dict_vector_to_np_array(serialized_representation['camera_pose']['location'])
            if 'rotation' in serialized_representation['camera_pose']:
                rot = geom.dict_quaternion_to_np_array(serialized_representation['camera_pose']['rotation'])
            if loc is not None or rot is not None:
                kwargs['camera_pose'] =  tf.Transform(location=loc, rotation=rot, w_first=False)
        if 'additional_metadata' in serialized_representation:
            kwargs['additional_metadata'] = serialized_representation['additional_metadata']
        if 'depth_filename' in serialized_representation:
            kwargs['depth_filename'] = serialized_representation['depth_filename']
        if 'labels_filename' in serialized_representation:
            kwargs['labels_filename'] = serialized_representation['labels_filename']
        if 'world_normals_filename' in serialized_representation:
            kwargs['world_normals_filename'] = serialized_representation['world_normals_filename']
        return super().deserialize(serialized_representation, **kwargs)

    @classmethod
    def from_image(cls, image):
        return cls(timestamp=image.timestamp,
                   filename=image.filename,
                   camera_location=image.camera_location,
                   camera_orientation=image.camera_orientation,
                   additional_metadata=image.additional_metadata,
                   depth_filename=image.depth_filename,
                   labels_filename=image.labels_filename,
                   world_normals_filename=image.world_normals_filename)


class StereoImageEntity(core.image.StereoImage, ImageEntity):

    def validate(self):
        if not super().validate():
            return False
        if not os.path.isfile(self.right_filename):
            return False
        if self.depth_filename is not None and not os.path.isfile(self.right_depth_filename):
            return False
        if self.labels_filename is not None and not os.path.isfile(self.right_labels_filename):
            return False
        if self.depth_filename is not None and not os.path.isfile(self.right_depth_filename):
            return False
        return True

    def serialize(self):
        serialized = super().serialize()

        # fiddle the serialized version for left and right images
        fiddle_keys = ['filename', 'camera_pose', 'depth_filename', 'labels_filename', 'world_normals_filename']
        for key in fiddle_keys:
            serialized['left_' + key] = serialized[key]
            del serialized[key]

        serialized['right_filename'] = self.right_filename
        serialized['right_camera_pose'] = {
            'location': geom.numpy_vector_to_dict(self.right_camera_location),
            'rotation': geom.numpy_quarternion_to_dict(self.right_camera_orientation)
        }
        serialized['right_depth_filename'] = self.right_depth_filename
        serialized['right_labels_filename'] = self.right_labels_filename
        serialized['right_world_normals_filename'] = self.right_world_normals_filename
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):

        if 'left_filename' in serialized_representation:
            kwargs['left_filename'] = serialized_representation['left_filename']
        if 'left_camera_pose' in serialized_representation:
            loc = None
            rot = None
            if 'location' in serialized_representation['left_camera_pose']:
                loc = geom.dict_vector_to_np_array(serialized_representation['left_camera_pose']['location'])
            if 'rotation' in serialized_representation['left_camera_pose']:
                rot = geom.dict_quaternion_to_np_array(serialized_representation['left_camera_pose']['rotation'])
            if loc is not None or rot is not None:
                kwargs['left_camera_pose'] =  tf.Transform(location=loc, rotation=rot, w_first=False)
        if 'left_depth_filename' in serialized_representation:
            kwargs['left_depth_filename'] = serialized_representation['left_depth_filename']
        if 'left_labels_filename' in serialized_representation:
            kwargs['left_labels_filename'] = serialized_representation['left_labels_filename']
        if 'left_world_normals_filename' in serialized_representation:
            kwargs['left_world_normals_filename'] = serialized_representation['left_world_normals_filename']

        if 'right_filename' in serialized_representation:
            kwargs['right_filename'] = serialized_representation['right_filename']
        if 'right_camera_pose' in serialized_representation:
            loc = None
            rot = None
            if 'location' in serialized_representation['right_camera_pose']:
                loc = geom.dict_vector_to_np_array(serialized_representation['right_camera_pose']['location'])
            if 'rotation' in serialized_representation['right_camera_pose']:
                rot = geom.dict_quaternion_to_np_array(serialized_representation['right_camera_pose']['rotation'])
            if loc is not None or rot is not None:
                kwargs['right_camera_pose'] =  tf.Transform(location=loc, rotation=rot, w_first=False)
        if 'right_depth_filename' in serialized_representation:
            kwargs['right_depth_filename'] = serialized_representation['right_depth_filename']
        if 'right_labels_filename' in serialized_representation:
            kwargs['right_labels_filename'] = serialized_representation['right_labels_filename']
        if 'right_world_normals_filename' in serialized_representation:
            kwargs['right_world_normals_filename'] = serialized_representation['right_world_normals_filename']
        return super().deserialize(serialized_representation, **kwargs)
