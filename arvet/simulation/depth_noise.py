# Copyright (c) 2017, John Skinner
import numpy as np
import arvet.util.image_utils as image_utils
import arvet.util.transform as tf
import arvet.metadata.camera_intrinsics as cam_intr


MAXIMUM_QUALITY = 10


def generate_depth_noise(left_ground_truth_depth: np.ndarray, right_ground_truth_depth: np.ndarray,
                         camera_intrinsics: cam_intr.CameraIntrinsics,
                         right_camera_relative_pose: tf.Transform,
                         quality_level: int = MAXIMUM_QUALITY) -> np.ndarray:
    """
    Generate a noisy depth image from a ground truth depth image.
    The image should already be float32 and scaled to meters.
    The output image will be the same size as the ground truth image, with noise introduced.
    :param left_ground_truth_depth: A ground-truth depth image captured from the simulator
    :param right_ground_truth_depth: A ground-truth depth image captured to the right of the main image.
    :param camera_intrinsics: The intrinsics for both cameras, assumed to be the same
    :param right_camera_relative_pose: The relative pose of the right camera, for projection logic.
    :param quality_level: An integer switch between different noise models, lower is worse. Default best model.
    :return:
    """
    quality_level = int(quality_level)
    if quality_level <= 0:
        return naive_gaussian_noise(left_ground_truth_depth)
    else:
        return kinect_depth_model(left_ground_truth_depth, right_ground_truth_depth, camera_intrinsics,
                                  right_camera_relative_pose)


def naive_gaussian_noise(ground_truth_depth: np.ndarray, variance: float = 0.1) -> np.ndarray:
    """
    The simplest and least realistic depth noise, we add a gaussian noise to each pixel.
    Noise is variance 0.1
    :param ground_truth_depth: Ground truth depth image
    :param variance: The variance in the ground-truth noise
    :return:
    """
    return ground_truth_depth + np.random.normal(0, variance, ground_truth_depth.shape)


def kinect_depth_model(ground_truth_depth_left: np.ndarray, ground_truth_depth_right: np.ndarray,
                       camera_intrinsics: cam_intr.CameraIntrinsics, baseline: tf.Transform) -> np.ndarray:
    """
    Depth noise based on the original kinect depth sensor
    :param ground_truth_depth_left: The left depth image
    :param ground_truth_depth_right: The right depth image
    :param camera_intrinsics: The intrinsics of both cameras
    :param baseline: The location of the right camera relative to the left camera
    :return:
    """
    # Coordinate transform baseline into camera coordinates, X right, Y down, Z forward
    # We only actually use the X any Y relative coordinates, assume the Z (forward) relative coordinate
    # is embedded in the right ground truth depth, i.e.: right_gt_depth = B_z + left_gt_depth (if B_x and B_y are zero)
    baseline_x = -1 * baseline.location[1]
    baseline_y = -1 * baseline.location[2]

    # Step 1: Rescale the camera intrisics to the kinect resolution
    fx = 640 * camera_intrinsics.fx / camera_intrinsics.width
    fy = 480 * camera_intrinsics.fy / camera_intrinsics.height
    cx = 640 * camera_intrinsics.cx / camera_intrinsics.width
    cy = 480 * camera_intrinsics.cy / camera_intrinsics.height

    # Step 2: Image resolution - kinect images are 640x480
    if ground_truth_depth_left.shape == (480, 640):
        left_depth_points = np.copy(ground_truth_depth_left)
    else:
        left_depth_points = image_utils.resize(ground_truth_depth_left, (640, 480),
                                               interpolation=image_utils.Interpolation.NEAREST)
    if ground_truth_depth_right.shape == (480, 640):
        right_depth_points = np.copy(ground_truth_depth_right)
    else:
        right_depth_points = image_utils.resize(ground_truth_depth_right, (640, 480),
                                                interpolation=image_utils.Interpolation.NEAREST)

    # Step 3: Find orthographic depth
    # Basically, we find the z-component of the world point for each depth point
    # d^2 = X^2 + Y^2 + Z^2
    # and
    # X = Z * (x - cx) / fx, Y = Z * (y - cy) / fy
    # Therefore:
    # Z = d / sqrt(((x - cx) / fx)^2 + ((y - cy) / fy)^2 + 1)
    # Z = d / |(x - cx) / fx, (y - cy) / fy, 1|
    ortho_projection = np.indices((480, 640), dtype=np.float32)
    ortho_projection = np.dstack((ortho_projection[1], ortho_projection[0], np.ones((480, 640), dtype=np.float32)))
    ortho_projection -= (cx, cy, 0)
    ortho_projection = np.divide(ortho_projection, (fy, fx, 1))  # Gives us (x - cx) / fx, (y - cy) / fy, 1
    ortho_projection = np.linalg.norm(ortho_projection, axis=2)
    output_depth = np.divide(left_depth_points, ortho_projection)

    # Step 4: Clipping planes - Set to 0 where too close or too far
    shadow_mask = (0.8 < output_depth) & (output_depth < 4.0)

    # Step 5: Shadows
    # Project the depth points from the right depth image onto the left depth image
    # Places that are not visible from the right image are shadows
    right_ortho_depth = np.divide(right_depth_points, ortho_projection)

    right_points = np.indices((480, 640), dtype=np.float32)
    right_x = right_points[1] - cx
    right_y = right_points[0] - cy

    # Stereo project points in right image into left image
    # x' = fx * (X - B_x) / (Z - B_z) + cx, y' = fy * (Y - B_y) / (Z - B_z) + cy
    # and, as above:
    # X = Z * (x - cx) / fx, Y = Z * (y - cy) / fy
    # Therefore,
    # x' = ((x - cx) * Z - fx * B_x) / (Z - B_z) + cx, similar for y
    right_x = np.multiply(output_depth, right_x)  # (x - cx) * Z
    right_y = np.multiply(output_depth, right_y)  # (y - cy) * Z
    # x * Z - fx * B_x, y * Z - fy * B_y
    right_x -= baseline_x * fx
    right_y -= baseline_y * fy
    # Divide throughout by Z - B_z, or just the orthographic right depth
    right_x = np.divide(right_x, right_ortho_depth + 0.00001) + cx
    right_y = np.divide(right_y, right_ortho_depth + 0.00001) + cy
    shadow_mask &= (right_x >= 0) & (right_y >= 0) & (right_x < 640) & (right_y < 480)
    projected_depth = nearest_sample(right_ortho_depth, right_x, right_y)
    shadow_mask &= (output_depth - projected_depth) < 0.01

    # Step 6: Random dropout of pixels
    shadow_mask &= np.random.choice([False, True], (480, 640), p=(0.2, 0.8))

    # Step 7: Lateral noise - I don't know how to do this quickly

    # Step 8: axial noise
    output_depth += np.random.normal(0, 0.0012 + 0.0019 * np.square(output_depth - 0.4))
    output_depth = np.multiply(shadow_mask, output_depth)

    # Finally, return to an image matching the input size, so that we're aligned with the RGB image
    if ground_truth_depth_left.shape != (480, 640):
        output_depth = image_utils.resize(output_depth, (ground_truth_depth_left.shape[1],
                                                         ground_truth_depth_left.shape[0]),
                                          interpolation=image_utils.Interpolation.NEAREST)
    return output_depth


def nearest_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample an image from fractional coordinates
    :param image:
    :param x:
    :param y:
    :return:
    """
    x = np.rint(x).astype(np.int)
    y = np.rint(y).astype(np.int)

    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)

    return image[y, x]


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Read values from image at the given x and y coordinates, using bilinaer sampling
    :param image: The source image to read from
    :param x: An image of x coordinates to read
    :param y: A similar image of y coordinates to read, must have the same shape as x
    :return: An image with the same shape as x and y, with values drawn from image
    """
    # Source: Alex Flint, https://stackoverflow.com/users/795053/alex-flint
    # Question: https://stackoverflow.com/questions/12729228/
    x0 = np.floor(x).astype(np.int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    im_a = image[y0, x0]
    im_b = image[y1, x0]
    im_c = image[y0, x1]
    im_d = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * im_a + wb * im_b + wc * im_c + wd * im_d
