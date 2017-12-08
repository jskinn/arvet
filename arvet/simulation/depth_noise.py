# Copyright (c) 2017, John Skinner
import numpy as np
import arvet.util.image_utils as image_utils


MAXIMUM_QUALITY = 10


def generate_depth_noise(left_ground_truth_depth, right_ground_truth_depth, camera_intrinsics,
                         right_camera_relative_pose, quality_level=MAXIMUM_QUALITY):
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
                                  right_camera_relative_pose.location)


def naive_gaussian_noise(ground_truth_depth, variance=0.1):
    """
    The simplest and least realistic depth noise, we add a gaussian noise to each pixel.
    Noise is variance 0.1
    :param ground_truth_depth: Ground truth depth image
    :param variance: The variance in the ground-truth noise
    :return:
    """
    return ground_truth_depth + np.random.normal(0, variance, ground_truth_depth.shape)


def kinect_depth_model(ground_truth_depth_left, ground_truth_depth_right, camera_intrinsics, baseline=(0, -0.15, 0)):
    """
    Depth noise based on the original kinect depth sensor
    :param ground_truth_depth_left: The left depth image
    :param ground_truth_depth_right: The right depth image
    :param camera_intrinsics: The intrinsics of both cameras
    :param baseline: The location of the right camera relative to the left camera
    :return:
    """
    # Coordinate transform baseline into camera coordinates, X right, Y down, Z forward
    # We want to project the baseline to the image plane, which we can't do if it's on the image plane, so we tweak z
    if isinstance(baseline, int) or isinstance(baseline, float):
        baseline = (baseline, 0, 0)
    elif np.array_equal(baseline, (0, 0, 0)):
        baseline = (0.15, 0, 0)
    elif baseline[0] != 0:
        raise ValueError("We cannot process stereo images where the stereo pair is not in the same plane.")
    else:
        baseline = (-baseline[1], -baseline[2], baseline[0])

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
    ortho_projection = np.array([[(x, y, 1) for x in range(640)] for y in range(480)], dtype=np.float32)
    ortho_projection = np.linalg.norm(np.divide(ortho_projection - (cx, cy, 0), (fx, fy, 1)), axis=2)
    output_depth = np.multiply(left_depth_points, ortho_projection)

    # Step 4: Clipping planes - Set to 0 where too close or too far
    shadow_mask = (output_depth > 0.8) & (output_depth < 4.0)

    # Step 5: Shadows
    # Project the depth points from the right depth image onto the left depth image
    # Places that are not visible from the right image are shadows
    right_ortho_depth = np.multiply(right_depth_points, ortho_projection)

    # TODO: This is 40% of the compute time. use np.indices((480, 640)) instead
    right_points = (np.array([[(x, y) for x in range(640)] for y in range(480)], dtype=np.float32)
                    - np.divide((baseline[0] * fx, baseline[1] * fy),
                                np.stack([right_ortho_depth + 0.00001, right_ortho_depth + 0.00001], axis=2)))
    right_points = np.asarray(np.rint(right_points), dtype=np.int32)    # TODO: This needs to bilinear sample
    shadow_mask &= np.all(right_points > (0, 0), axis=2) & np.all(right_points < (640, 480), axis=2)
    projected_depth = right_ortho_depth[right_points[:, :, 1], right_points[:, :, 0]]
    shadow_mask &= (output_depth - projected_depth) < 0.01

    # Step 6: Random dropout
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
