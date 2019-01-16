# Copyright (c) 2017, John Skinner
from operator import itemgetter
import pymodm

from arvet.database.enum_field import EnumField
from arvet.database.transform_field import TransformField
import arvet.metadata.camera_intrinsics as cam_intr
from arvet.metadata.image_metadata import MaskedObject
from arvet.core.image import Image, StereoImage
from arvet.core.sequence_type import ImageSequenceType
import arvet.core.image_source


class ImageCollection(arvet.core.image_source.ImageSource, pymodm.MongoModel):
    """
    A collection of images stored in the database.
    This can be a sequential set of images like a video, or a random sampling of different pictures.
    """
    images = pymodm.fields.ListField(
        pymodm.ReferenceField(Image, required=True, on_delete=pymodm.fields.ReferenceField.CASCADE),
        required=True
    )
    timestamps = pymodm.fields.ListField(pymodm.fields.FloatField(required=True), required=True)
    sequence_type = EnumField(ImageSequenceType, required=True)

    is_depth_available = pymodm.fields.BooleanField(required=True)
    is_normals_available = pymodm.fields.BooleanField(required=True)
    is_stereo_available = pymodm.fields.BooleanField(required=True)
    is_labels_available = pymodm.fields.BooleanField(required=True)
    is_masks_available = pymodm.fields.BooleanField(required=True)
    is_stored_in_database = True

    camera_intrinsics = pymodm.fields.EmbeddedDocumentField(cam_intr.CameraIntrinsics, required=True)
    right_camera_pose = TransformField()
    right_camera_intrinsics = pymodm.fields.EmbeddedDocumentField(cam_intr.CameraIntrinsics)

    def __init__(self, *args, **kwargs):
        if (len(args) >= 3 and args[2] is ImageSequenceType.INTERACTIVE) or \
                ('sequence_type' in kwargs and kwargs['sequence_type'] is ImageSequenceType.INTERACTIVE):
            raise ValueError("Image Collections cannot be interactive")
        images = args[0] if len(args) >= 1 else kwargs.get('images', None)
        timestamps = args[1] if len(args) >= 2 else kwargs.get('timestamps', None)
        if images is not None and timestamps is not None:
            if not len(images) == len(timestamps):
                raise ValueError("The number of images must match the number of timestamps")

            # Sort the images and timestamps together
            pairs = sorted(zip(timestamps, images), key=itemgetter(0))
            images = [pair[1] for pair in pairs]
            timestamps = [pair[0] for pair in pairs]
            if len(args) > 2:
                args = (images, timestamps) + args[2:]
            else:
                args = ()
                kwargs['images'] = images
                kwargs['timestamps'] = timestamps

            # Derive all the other properties if they are unspecified
            if len(args) < 4 and 'is_depth_available' not in kwargs:
                kwargs['is_depth_available'] = all(image.depth is not None for image in images)
            if len(args) < 5 and 'is_normals_available' not in kwargs:
                kwargs['is_normals_available'] = all(image.normals is not None for image in images)
            if len(args) < 6 and 'is_stereo_available' not in kwargs:
                kwargs['is_stereo_available'] = all(isinstance(image, StereoImage) for image in images)
            if len(args) < 7 and 'is_labels_available' not in kwargs:
                kwargs['is_labels_available'] = any(len(image.metadata.labelled_objects) > 0 for image in images)
            if len(args) < 8 and 'is_masks_available' not in kwargs:
                kwargs['is_masks_available'] = all(
                    any(isinstance(label, MaskedObject) for label in image.metadata.labelled_objects)
                    for image in images)
            if len(images) > 0:
                first_image = images[0]
                if len(args) < 9 and 'camera_intrinsics' not in kwargs:
                    kwargs['camera_intrinsics'] = first_image.metadata.intrinsics
                if isinstance(first_image, StereoImage):
                    if len(args) < 10 and 'right_camera_pose' not in kwargs:
                        kwargs['right_camera_pose'] = first_image.left_camera_pose.find_relative(
                            first_image.right_camera_pose)
                    if len(args) < 11 and 'right_camera_intrinsics' not in kwargs:
                        kwargs['right_camera_intrinsics'] = first_image.right_metadata.intrinsics
        super().__init__(*args, **kwargs)

    def __len__(self):
        """
        The length of the image collection
        :return:
        """
        return len(self.images)

    def __iter__(self):
        """
        Iterator for the image collection.
        Returns a timestamp and image for each iteration
        :return:
        """
        for timestamp, image in sorted(zip(self.timestamps, self.images), key=itemgetter(0)):
            yield timestamp, image

    def __getitem__(self, item):
        """
        Allow index-based access. Why not.
        This is the same as get
        :param item:
        :return:
        """
        return self.timestamps[item], self.images[item]
