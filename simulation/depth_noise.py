# Copyright (c) 2017, John Skinner
import functools
import numpy as np
import cv2


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
        left_depth_points = cv2.resize(ground_truth_depth_left, (640, 480), interpolation=cv2.INTER_NEAREST)
    if ground_truth_depth_right.shape == (480, 640):
        right_depth_points = np.copy(ground_truth_depth_right)
    else:
        right_depth_points = cv2.resize(ground_truth_depth_right, (640, 480), interpolation=cv2.INTER_NEAREST)

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
    projected_depth = right_ortho_depth[right_points[:, :, 1], right_points[:,:,0]]
    shadow_mask &= (output_depth - projected_depth) < 0.01

    # Step 6: Random dropout
    shadow_mask &= np.random.choice([False, True], (480, 640), p=(0.2, 0.8))

    # Step 7: Lateral noise - I don't know how to do this quickly

    # Step 8: axial noise
    output_depth += np.random.normal(0, 0.0012 + 0.0019 * np.square(output_depth - 0.4))
    output_depth = np.multiply(shadow_mask, output_depth)

    # Finally, return to an image matching the input size, so that we're aligned with the RGB image
    if ground_truth_depth_left.shape != (480, 640):
        output_depth = cv2.resize(output_depth, (ground_truth_depth_left.shape[1], ground_truth_depth_left.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    return output_depth


# --------- Code Graveyard ---------


def __kinect_depth_model_raycast(ground_truth_depth, camera_intrinsics, baseline=(0.00001, -0.15, 0)):
    """
    Depth noise based on the original kinect depth sensor
    This version raycasts from a baseline point for each pixel.
    Unusably slow.
    :param ground_truth_depth:
    :return:
    """
    # Coordinate transform baseline into camera coordinates, X right, Y down, Z forward
    # We want to project the baseline to the image plane, which we can't do if it's on the image plane, so we tweak z
    if isinstance(baseline, int) or isinstance(baseline, float):
        baseline = (baseline, 0, 0.00001)
    elif np.array_equal(baseline, (0, 0, 0)):
        baseline = (0.15, 0, 0.00001)
    elif baseline[0] <= 0:
        baseline = (-baseline[1], -baseline[2], 0.00001)
    else:
        baseline = (-baseline[1], -baseline[2], baseline[0])


    # Step 1: Image resolution - kinect images are 640x480
    if ground_truth_depth.shape == (480, 640):
        output_depth = np.copy(ground_truth_depth)
    else:
        output_depth = cv2.resize(ground_truth_depth, (640, 480), interpolation=cv2.INTER_CUBIC)

    # Step 2: Rescale the camera intrisics to the new resolution
    fx = 640 * camera_intrinsics.fx
    fy = 480 * camera_intrinsics.fy
    cx = 640 * camera_intrinsics.cx
    cy = 480 * camera_intrinsics.cy

    # Project the world position of the pattern projector onto the image plane, it should be outside the image bounds
    projection_source = (fx * baseline[0] / baseline[2] + cx, fy * baseline[1] / baseline[2] + cy)

    # Steps 3-5: The next steps are done per-pixel
    do_transform = functools.partial(__transform_pixel_raycast, fx, fy, cx, cy, ground_truth_depth, baseline, projection_source)
    output_depth = np.fromiter((do_transform(x, y) for y in range(480) for x in range(640)),
                               dtype=np.float32, count=640*480)
    output_depth = output_depth.reshape((480, 640))

    # Step 6: Lateral noise -

    # Step 7: axial noise
    zero_mask = output_depth == 0
    output_depth += zero_mask * np.random.normal(0, 0.0012 + 0.0019 * np.square(output_depth - 0.4))

    # Finally, return to an image matching the input size, so that we're aligned with the RGB image
    if ground_truth_depth.shape != (480, 640):
        output_depth = cv2.resize(output_depth, (ground_truth_depth.shape[1], ground_truth_depth.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
    return output_depth


def __transform_pixel_raycast(fx, fy, cx, cy, depth_source, projection_world_point, projection_image_point, x, y):
    """
    Per-pixel operations for checking if a pixel is shadowed
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :param depth_source:
    :param projection_world_point:
    :param projection_image_point:
    :param x:
    :param y:
    :return:
    """
    # Step 3: Convert from perspective to orthographic depth
    world_point = np.array(((x - cx) / fx, (y - cy) / fy, 1))   # This is the same as (X/Z, Y/Z, 1)
    # We don't know Z, but we know |X, Y, Z|, rescale
    world_point = __get_depth_at_point(x, y, depth_source) * world_point / np.linalg.norm(world_point)
    # The Z component of world_point is now our orthographic depth, corrected for perspective

    # Step 4: Clipping planes - Set to nan where too close or too far
    if world_point[2] < 0.4 or world_point[2] > 4.0:
        return 0.0
    else:
        # Step 5: Shadows - points that are behind other points relative to the projector off to the right are 0.
        # Precalculate some values for estimating a world point on the line (world_point -> projection_world_point)
        x_ratio = y_ratio = 0
        x_numerator = y_numerator = 1
        if world_point[0] != projection_world_point[0]:
            x_ratio = (projection_world_point[2] - world_point[2]) / (projection_world_point[0] - world_point[0])
            x_numerator = world_point[2] - world_point[0] * x_ratio
        if world_point[1] != projection_world_point[1]:
            y_ratio = (projection_world_point[2] - world_point[2]) / (projection_world_point[1] - world_point[1])
            y_numerator = world_point[2] - world_point[1] * y_ratio

        search_pixel = np.array((x, y), dtype=np.float32)
        step = projection_image_point - search_pixel
        # This makes the largest step direction 1, so we hit each pixel in our fastest changing direction
        step /= np.max(np.abs(step))
        search_pixel += step    # Skip the starting pixel
        while 0 <= search_pixel[0] < 640 and 0 <= search_pixel[1] < 480 and not (
                    search_pixel[0] == projection_image_point[0] and search_pixel[1] == projection_image_point[1]):
            observed_depth = __get_depth_at_point(search_pixel[0], search_pixel[1], depth_source)

            # Estimate the point on the ray from this search pixel
            # We have 4 different ways of doing this, over either x or y
            # It seems like there should be a geometric explanation for how this works, but honestly,
            # I just solved the intersection of the inverse pixel projection with the known interpolation between
            # world point and the projection source
            x_per_z = (search_pixel[0] - cx) / fx
            y_per_z = (search_pixel[1] - cy) / fy
            z = None
            if world_point[0] != projection_world_point[0] and x_per_z * x_ratio != 1:
                z = x_numerator / (1 - x_per_z * x_ratio)
            elif world_point[0] == projection_world_point[0] and x_per_z != 0:
                z = world_point[0] / x_per_z
            elif world_point[1] != projection_world_point[1] and not y_per_z * y_ratio != 1:
                z = y_numerator / (1 - y_per_z * y_ratio)
            elif world_point[1] == projection_world_point[1] and y_per_z != 0:
                z = world_point[1] / y_per_z
            if z is not None and observed_depth < np.linalg.norm((x_per_z * z, y_per_z * z, z)):
                # This pixel is obscured, return 0 reading
                return 0.0
            search_pixel += step

        # Pixel is not in shadow, return the orthographic depth
        return world_point[2]


def __kinect_depth_stereo_model(ground_truth_depth_left, ground_truth_depth_right, camera_intrinsics, baseline=(0, -0.15, 0)):
    """
    Depth noise based on the original kinect depth sensor
    First-cut stereo version, still performs per-pixel comparisons.
    Is still too slow, though not by much.

    :param ground_truth_depth_left:
    :return:
    """
    # Coordinate transform baseline into camera coordinates, X right, Y down, Z forward
    # We want to project the baseline to the image plane, which we can't do if it's on the image plane, so we tweak z
    if isinstance(baseline, int) or isinstance(baseline, float):
        baseline = (baseline, 0, 0)
    elif np.array_equal(baseline, (0, 0, 0)):
        baseline = (0.15, 0, 0)
    else:
        baseline = (-baseline[1], -baseline[2], baseline[0])

    # Step 1: Rescale the camera intrisics to the kinect resolution
    fx = 640 * camera_intrinsics.fx
    fy = 480 * camera_intrinsics.fy
    cx = 640 * camera_intrinsics.cx
    cy = 480 * camera_intrinsics.cy

    # Steps 2-6: Sample the depth maps per-pixel
    do_transform = functools.partial(__transform_pixel_stereo, fx, fy, cx, cy, ground_truth_depth_left,
                                     ground_truth_depth_right, baseline[0], baseline[1])
    do_transform = np.vectorize(do_transform, otypes=(np.float32,))
    output_depth = np.fromfunction(do_transform, (480, 640))


    # Finally, return to an image matching the input size, so that we're aligned with the RGB image
    if ground_truth_depth_left.shape != (480, 640):
        output_depth = cv2.resize(output_depth, (ground_truth_depth_left.shape[1], ground_truth_depth_left.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
    return output_depth


def __transform_pixel_stereo(fx, fy, cx, cy, depth_source, right_depth_source, baseline_x, baseline_y, y, x):
    # Step 2: Get the true depth at the pixel, used for noise and p
    true_depth = __get_depth_at_point(x, y, depth_source)

    # Step 3: Find the world point from the true depth
    world_point = np.array(((x - cx) / fx, (y - cy) / fy, 1))   # This is the same as (X/Z, Y/Z, 1)
    # We don't know Z, but we know |X, Y, Z|, rescale
    world_point = true_depth * world_point / np.linalg.norm(world_point)
    # The Z component of world_point is now our orthographic depth, corrected for perspective

    # Step 4: Clipping planes - Set to nan where too close or too far
    if world_point[2] < 0.4 or world_point[2] > 4.0:
        return 0.0

    # Step 5: Shadows - points that are behind other points relative to the projector off to the right are 0.
    # Find the matching pixel in the right image
    x2 = x - baseline_x * fx / world_point[2]
    y2 = y - baseline_y * fy / world_point[2]
    if x2 < 0 or x2 > 640 or y2 < 0 or y2 > 480 or __get_depth_at_point(x2, y2, right_depth_source) < true_depth:
        # Point is not visible from the right camera, its in shadow
        return 0.0

    # Step 6: Lateral noise -
    return world_point[2]

    # Step 7: axial noise
    #output_depth += zero_mask * np.random.normal(0, 0.0012 + 0.0019 * np.square(output_depth - 0.4))


def __get_depth_at_point(x, y, depth_source):
    """
    Get a sub-pixel sample from an image.
    This is a big candidate for why the two earlier variants are so slow.
    Indexes are always as for a 640x480 image, even if the source image is a different size.
    :param x: x-coordinate to sample, in range 0 <= x < 640
    :param y: y-coordinage to sample, in range 0 <= y < 480
    :param depth_source: The depth image to sample from
    :return: A depth value at the given coordinate
    """
    # Take a sample from a larger depth image, using bilinear interpolation
    source_x = depth_source.shape[1] * x / 640
    source_y = depth_source.shape[0] * y / 480

    x0 = int(source_x)
    y0 = int(source_y)
    rx = source_x - x0
    ry = source_y - y0
    x1 = min(x0 + 1, depth_source.shape[1] - 1)
    y1 = min(y0 + 1, depth_source.shape[0] - 1)
    return ((depth_source[y0, x0] * (1 - rx) + depth_source[y0, x1] * rx) * (1 - ry) +
            (depth_source[y1, x0] * (1 - rx) + depth_source[y1, x1] * rx) * ry)
