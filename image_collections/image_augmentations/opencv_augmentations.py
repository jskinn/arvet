#Copyright (c) 2017, John Skinner
import numpy as np
import cv2
import xxhash
import core.image
import metadata.image_metadata as imeta
import image_collections.augmented_collection as aug_coll


class Rotate(aug_coll.ImageAugmenter):
    """
    Rotate the image about a point, by an angle.
    If you're going to do a 90 degree rotation, we have simple augmenters for that.
    """

    def __init__(self, theta, origin_x=0.5, origin_y=0.5, id_=None):
        """
        Create the augmenter
        :param theta: The angle to rotate the image by, in radians
        :param origin_x: The x-coordinate of the point to rotate the image around, as a fraction of the image width.
        Defaults to 0.5, the image centre.
        :param origin_y: The y-coordinate of the point to rotate the image around, as a fraction of the image width.
        Defaults to 0.5, the image centre.
        """
        super().__init__(id_=id_)
        self._theta = theta
        self._origin_x = origin_x
        self._origin_y = origin_y

    def augment(self, image):
        transformation_matrix = np.identity(3)
        origin = (self._origin_x * (image.data.shape[1] - 1), self._origin_y * (image.data.shape[0] - 1))
        transformation_matrix[0:2, :] = cv2.getRotationMatrix2D(origin, 180 * self._theta / np.pi, 1)
        return warp_image(image, transformation_matrix)


class Translate(aug_coll.ImageAugmenter):
    """
    Translate the image by a fraction of its resolution
    """

    def __init__(self, translation_x, translation_y, id_=None):
        """
        Create the augmenter
        :param translation_x: How much to change the x image coordinate (left positive, up negative),
        as a fraction of the image resolution
        :param translation_y: How much to change the y image coordinate (down positive, up negative),
        as a fraction of the image resolution
        """
        super().__init__(id_=id_)
        self._translation_x = translation_x
        self._translation_y = translation_y

    def augment(self, image):
        transformation_matrix = np.identity(3)
        transformation_matrix[0, 2] = self._translation_x * image.data.shape[1]
        transformation_matrix[1, 2] = self._translation_y * image.data.shape[0]
        return warp_image(image, transformation_matrix)


class WarpAffine(aug_coll.ImageAugmenter):
    """
    Form an affine warp using 3 pairs of points, mapping from original image to warped image.
    To handle different resolutions, image points are expressed as fractions of the image size.
    That is, points are range [0-1), [0-1), which are then multiplied by the actual
    resolution of the image.

    There are simple augmentations implemented for a lot of simple changes like flipping or
    90 degree rotations, which are more efficient, use those instead where possible
    """

    def __init__(self, point1, point1_prime, point2, point2_prime, point3, point3_prime, id_=None):
        super().__init__(id_=id_)
        self._point1 = point1
        self._point1_prime = point1_prime
        self._point2 = point2
        self._point2_prime = point2_prime
        self._point3 = point3
        self._point3_prime = point3_prime

    def augment(self, image):
        resolution = np.array((image.data.shape[1], image.data.shape[0]))
        original_points = np.array([
            resolution * self._point1,
            resolution * self._point2,
            resolution * self._point3
        ], dtype=np.float32)
        warped_points = np.array([
            resolution * self._point1_prime,
            resolution * self._point2_prime,
            resolution * self._point3_prime
        ], dtype=np.float32)
        transformation_matrix = np.identity(3)
        transformation_matrix[0:2, :] = cv2.getAffineTransform(original_points, warped_points)
        return warp_image(image, transformation_matrix)


def warp_image(image, transformation_matrix):
    # Warp the base image
    transformed_data = cv2.warpAffine(np.asarray(image.data, dtype=np.float), transformation_matrix[0:2, :],
                                      image.data.shape[0:2], flags=cv2.INTER_CUBIC)
    transformed_data = np.asarray(transformed_data, dtype='uint8')

    # Warp the image labels if available
    if image.labels_data is not None:
        transformed_labels = cv2.warpAffine(np.asarray(image.labels_data, dtype=np.float),
                                            transformation_matrix[0:2, :], image.data.shape[0:2], flags=cv2.INTER_CUBIC)
        transformed_labels = np.asarray(transformed_labels, dtype='uint8')
    else:
        transformed_labels = None

    # Warp the depth image if available
    if image.depth_data is not None:
        transformed_depth = cv2.warpAffine(np.asarray(image.depth_data, dtype=np.float), transformation_matrix[0:2, :],
                                           image.data.shape[0:2], flags=cv2.INTER_CUBIC)
        transformed_depth = np.asarray(transformed_depth, dtype='uint8')
    else:
        transformed_depth = None

    # Warp the world normals if available
    if image.world_normals_data is not None:
        transformed_normals = cv2.warpAffine(np.asarray(image.world_normals_data, dtype=np.float),
                                             transformation_matrix[0:2, :], image.data.shape[0:2],
                                             flags=cv2.INTER_CUBIC)
        transformed_normals = np.asarray(transformed_normals, dtype='uint8')
    else:
        transformed_normals = None

    return core.image.Image(
        data=transformed_data,
        metadata=image.metadata.clone(
            hash_=xxhash.xxh64(transformed_data).digest(),
            base_image=image.metadata.base_image if image.metadata.base_image is not None else image,
            transformation_matrix=(np.dot(transformation_matrix, image.metadata.affine_transformation_matrix)
                                   if image.metadata.affine_transformation_matrix is not None
                                   else transformation_matrix),
            labelled_objects=transform_bounding_boxes(image.metadata.labelled_objects, transformation_matrix)),
        additional_metadata=image.additional_metadata,
        depth_data=transformed_depth,
        labels_data=transformed_labels,
        world_normals_data=transformed_normals
    )


def transform_bounding_boxes(labelled_objects, transformation_matrix):
    """
    Transform the labelled objects so that the bounding boxes match after the image transformation.
    :param labelled_objects: The list of labelled objects from the metadata
    :param transformation_matrix: The 3x3 affine transformation matrix used to modify the image.
    :return: A new list of labelled objects, with modified bounding boxes.
    """
    transformed_objects = []
    for obj in labelled_objects:
        upper_left = (obj.bounding_box[0], obj.bounding_box[1], 1)
        upper_right = (obj.bounding_box[0] + obj.bounding_box[2], obj.bounding_box[1], 1)
        lower_left = (obj.bounding_box[0], obj.bounding_box[1] + obj.bounding_box[3], 1)
        lower_right = (obj.bounding_box[0] + obj.bounding_box[2], obj.bounding_box[1] + obj.bounding_box[3], 1)

        # Project all the corners of the bounding box into the new image
        # Rounding to the nearest integer to avoid floating point errors (pixels are discrete)
        upper_left = np.round(np.dot(transformation_matrix, upper_left))
        upper_right = np.round(np.dot(transformation_matrix, upper_right))
        lower_left = np.round(np.dot(transformation_matrix, lower_left))
        lower_right = np.round(np.dot(transformation_matrix, lower_right))

        # Wrap an axis-aligned bounding box around the projected box
        new_upper_left = (min(upper_left[0], upper_right[0], lower_left[0], lower_right[0]),
                          min(upper_left[1], upper_right[1], lower_left[1], lower_right[1]))
        new_lower_right = (max(upper_left[0], upper_right[0], lower_left[0], lower_right[0]),
                           max(upper_left[1], upper_right[1], lower_left[1], lower_right[1]))

        transformed_objects.append(imeta.LabelledObject(
            class_names=obj.class_names,
            bounding_box=(
                new_upper_left[0],
                new_upper_left[1],
                new_lower_right[0] - new_upper_left[0],
                new_lower_right[1] - new_upper_left[1]
            ),
            label_color=obj.label_color,
            relative_pose=obj.relative_pose,
            object_id=obj.object_id
        ))
    return transformed_objects
