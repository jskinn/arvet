# Copyright (c) 2017, John Skinner
import numpy as np
import xxhash
import argus.core.image
import argus.metadata.image_metadata as imeta
import argus.image_collections.augmented_collection as aug_coll


class HorizontalFlip(aug_coll.ImageAugmenter):
    """
    An augmenter that changes an image by mirroring it horizontally,
    swapping left and right
    """
    def __init__(self, id_=None):
        super().__init__(id_=id_)

    def augment(self, image):
        """
        Augment the image
        :param image:
        :return:
        """
        flipped_data = np.fliplr(image.data)
        transformation_matrix = np.array([[-1, 0, image.data.shape[1] - 1],
                                          [0, 1, 0],
                                          [0, 0, 1]])
        if image.metadata.affine_transformation_matrix is not None:
            transformation_matrix = np.dot(transformation_matrix, image.metadata.affine_transformation_matrix)
        return argus.core.image.Image(
            data=flipped_data,
            metadata=image.metadata.clone(
                hash_=xxhash.xxh64(np.ascontiguousarray(flipped_data)).digest(),
                intrinsics=image.metadata.camera_intrinsics,
                base_image=image.metadata.base_image if image.metadata.base_image is not None else image,
                transformation_matrix=transformation_matrix,
                labelled_objects=(imeta.LabelledObject(
                    class_names=obj.class_names,
                    bounding_box=(
                        image.metadata.width - 1 - obj.bounding_box[0] - obj.bounding_box[2],
                        obj.bounding_box[1],
                        obj.bounding_box[2],
                        obj.bounding_box[3]
                    ),
                    label_color=obj.label_color,
                    relative_pose=obj.relative_pose,
                    object_id=obj.object_id)
                    for obj in image.metadata.labelled_objects)),
            additional_metadata=image.additional_metadata,
            depth_data=np.fliplr(image.depth_data) if image.depth_data is not None else None,
            labels_data=np.fliplr(image.labels_data) if image.labels_data is not None else None,
            world_normals_data=np.fliplr(image.world_normals_data) if image.world_normals_data is not None else None
        )


class VerticalFlip(aug_coll.ImageAugmenter):
    """
    An augmenter that changes an image by mirroring it vertically,
    swapping up and down
    """
    def __init__(self, id_=None):
        super().__init__(id_=id_)

    def augment(self, image):
        flipped_data = np.flipud(image.data)
        transformation_matrix = np.array([[1, 0, 0],
                                          [0, -1, image.data.shape[0] - 1],
                                          [0, 0, 1]])
        if image.metadata.affine_transformation_matrix is not None:
            transformation_matrix = np.dot(transformation_matrix, image.metadata.affine_transformation_matrix)
        return argus.core.image.Image(
            data=flipped_data,
            metadata=image.metadata.clone(
                hash_=xxhash.xxh64(np.ascontiguousarray(flipped_data)).digest(),
                intrinsics=image.metadata.camera_intrinsics,
                base_image=image.metadata.base_image if image.metadata.base_image is not None else image,
                transformation_matrix=transformation_matrix,
                labelled_objects=(imeta.LabelledObject(
                    class_names=obj.class_names,
                    bounding_box=(
                        obj.bounding_box[0],
                        image.metadata.height - 1 - obj.bounding_box[1] - obj.bounding_box[3],
                        obj.bounding_box[2],
                        obj.bounding_box[3]
                    ),
                    label_color=obj.label_color,
                    relative_pose=obj.relative_pose,
                    object_id=obj.object_id)
                    for obj in image.metadata.labelled_objects)),
            additional_metadata=image.additional_metadata,
            depth_data=np.flipud(image.depth_data) if image.depth_data is not None else None,
            labels_data=np.flipud(image.labels_data) if image.labels_data is not None else None,
            world_normals_data=np.flipud(image.world_normals_data) if image.world_normals_data is not None else None
        )


class Rotate90(aug_coll.ImageAugmenter):
    """
    An augmenter that changes an image by rotating it 90 degrees anticlockwise
    """
    def __init__(self, id_=None):
        super().__init__(id_=id_)

    def augment(self, image):
        centre_x = (image.data.shape[1] - 1) / 2
        centre_y = (image.data.shape[0] - 1) / 2
        rotated_data = np.rot90(image.data, k=1)
        transformation_matrix = np.array([[0, 1, centre_x - centre_y],
                                          [-1, 0, centre_x + centre_y],
                                          [0, 0, 1]])
        if image.metadata.affine_transformation_matrix is not None:
            transformation_matrix = np.dot(transformation_matrix, image.metadata.affine_transformation_matrix)
        return argus.core.image.Image(
            data=rotated_data,
            metadata=image.metadata.clone(
                hash_=xxhash.xxh64(np.ascontiguousarray(rotated_data)).digest(),
                intrinsics=image.metadata.camera_intrinsics,
                base_image=image.metadata.base_image if image.metadata.base_image is not None else image,
                transformation_matrix=transformation_matrix,
                labelled_objects=(imeta.LabelledObject(
                    class_names=obj.class_names,
                    bounding_box=(
                        obj.bounding_box[1],
                        image.metadata.width - 1 - obj.bounding_box[0] - obj.bounding_box[2],
                        obj.bounding_box[3],
                        obj.bounding_box[2]
                    ),
                    label_color=obj.label_color,
                    relative_pose=obj.relative_pose,
                    object_id=obj.object_id)
                    for obj in image.metadata.labelled_objects)),
            additional_metadata=image.additional_metadata,
            depth_data=np.rot90(image.depth_data, k=1) if image.depth_data is not None else None,
            labels_data=np.rot90(image.labels_data, k=1) if image.labels_data is not None else None,
            world_normals_data=np.rot90(image.world_normals_data, k=1) if image.world_normals_data is not None else None
        )


class Rotate180(aug_coll.ImageAugmenter):
    """
    An augmenter that changes an image by rotating it 180 degrees
    """
    def __init__(self, id_=None):
        super().__init__(id_=id_)

    def augment(self, image):
        rotated_data = np.rot90(image.data, k=2)
        transformation_matrix = np.array([[-1, 0, image.data.shape[1] - 1],
                                          [0, -1, image.data.shape[0] - 1],
                                          [0, 0, 1]])
        if image.metadata.affine_transformation_matrix is not None:
            transformation_matrix = np.dot(transformation_matrix, image.metadata.affine_transformation_matrix)
        return argus.core.image.Image(
            data=rotated_data,
            metadata=image.metadata.clone(
                hash_=xxhash.xxh64(np.ascontiguousarray(rotated_data)).digest(),
                intrinsics=image.metadata.camera_intrinsics,
                base_image=image.metadata.base_image if image.metadata.base_image is not None else image,
                transformation_matrix=transformation_matrix,
                labelled_objects=(imeta.LabelledObject(
                    class_names=obj.class_names,
                    bounding_box=(
                        image.metadata.width - 1 - obj.bounding_box[0] - obj.bounding_box[2],
                        image.metadata.height - 1 - obj.bounding_box[1] - obj.bounding_box[3],
                        obj.bounding_box[2],
                        obj.bounding_box[3]
                    ),
                    label_color=obj.label_color,
                    relative_pose=obj.relative_pose,
                    object_id=obj.object_id)
                    for obj in image.metadata.labelled_objects)),
            additional_metadata=image.additional_metadata,
            depth_data=np.rot90(image.depth_data, k=2) if image.depth_data is not None else None,
            labels_data=np.rot90(image.labels_data, k=2) if image.labels_data is not None else None,
            world_normals_data=np.rot90(image.world_normals_data, k=2) if image.world_normals_data is not None else None
        )


class Rotate270(aug_coll.ImageAugmenter):
    """
    An augmenter that changes an image by rotating it 270 degrees anticlockwise,
    which is the same as rotating it 90 degrees clockwise
    """
    def __init__(self, id_=None):
        super().__init__(id_=id_)

    def augment(self, image):
        centre_x = (image.data.shape[1] - 1) / 2
        centre_y = (image.data.shape[0] - 1) / 2
        rotated_data = np.rot90(image.data, k=3)
        transformation_matrix = np.array([[0, -1, centre_x + centre_y],
                                          [1, 0, centre_y - centre_x],
                                          [0, 0, 1]])
        if image.metadata.affine_transformation_matrix is not None:
            transformation_matrix = np.dot(transformation_matrix, image.metadata.affine_transformation_matrix)
        return argus.core.image.Image(
            data=rotated_data,
            metadata=image.metadata.clone(
                hash_=xxhash.xxh64(np.ascontiguousarray(rotated_data)).digest(),
                intrinsics=image.metadata.camera_intrinsics,
                base_image=image.metadata.base_image if image.metadata.base_image is not None else image,
                transformation_matrix=transformation_matrix,
                labelled_objects=(imeta.LabelledObject(
                    class_names=obj.class_names,
                    bounding_box=(
                        image.metadata.height - 1 - obj.bounding_box[1] - obj.bounding_box[3],
                        obj.bounding_box[0],
                        obj.bounding_box[3],
                        obj.bounding_box[2]
                    ),
                    label_color=obj.label_color,
                    relative_pose=obj.relative_pose,
                    object_id=obj.object_id)
                    for obj in image.metadata.labelled_objects)),
            additional_metadata=image.additional_metadata,
            depth_data=np.rot90(image.depth_data, k=3) if image.depth_data is not None else None,
            labels_data=np.rot90(image.labels_data, k=3) if image.labels_data is not None else None,
            world_normals_data=np.rot90(image.world_normals_data, k=3) if image.world_normals_data is not None else None
        )
