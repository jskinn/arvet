# Copyright (c) 2017, John Skinner
import os.path
import unittest
import numpy as np
import timeit
import arvet.util.image_utils as image_utils
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.util.transform as tf
import arvet.simulation.depth_noise as depth_noise


class TestDepthNoise(unittest.TestCase):

    def test_gaussian_noise(self):
        ground_truth_depth = get_test_image('left')
        noisy_depth = depth_noise.naive_gaussian_noise(ground_truth_depth)

        diff = noisy_depth - ground_truth_depth
        self.assertEqual(diff.shape[0] * diff.shape[1], np.count_nonzero(diff))
        self.assertAlmostEqual(0, np.mean(diff), delta=0.1)
        self.assertAlmostEqual(0.1, np.std(diff), delta=0.05)
        # image_utils.show_image(noisy_depth / np.max(noisy_depth), 'test depth')

    def test_kinect_noise_maintains_type(self):
        ground_truth_depth_left = 256 * get_test_image('left')
        ground_truth_depth_right = 256 * get_test_image('right')

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            ground_truth_depth_left.shape[1],
            ground_truth_depth_left.shape[0],
            focal_length * ground_truth_depth_left.shape[1],
            focal_length * ground_truth_depth_left.shape[1],
            0.5 * ground_truth_depth_left.shape[1], 0.5 * ground_truth_depth_left.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))
        noisy_depth = depth_noise.kinect_depth_model(ground_truth_depth_left,
                                                     ground_truth_depth_right, camera_intrinsics, relative_pose)
        self.assertIsNotNone(noisy_depth)
        self.assertNotEqual(np.dtype, noisy_depth.dtype)

    def test_kinect_noise_works_when_not_640_by_480(self):
        ground_truth_depth_left = get_test_image('left')[0:64, 0:64]
        ground_truth_depth_right = get_test_image('right')[0:64, 0:64]

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            ground_truth_depth_left.shape[1],
            ground_truth_depth_left.shape[0],
            focal_length * ground_truth_depth_left.shape[1],
            focal_length * ground_truth_depth_left.shape[1],
            0.5 * ground_truth_depth_left.shape[1], 0.5 * ground_truth_depth_left.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))
        noisy_depth = depth_noise.kinect_depth_model(ground_truth_depth_left,
                                                     ground_truth_depth_right, camera_intrinsics, relative_pose)
        self.assertIsNotNone(noisy_depth)
        self.assertNotEqual(np.dtype, noisy_depth.dtype)

    def test_kinect_noise_produces_reasonable_output(self):
        ground_truth_depth_left = get_test_image('left')
        ground_truth_depth_right = get_test_image('right')

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            ground_truth_depth_left.shape[1],
            ground_truth_depth_left.shape[0],
            focal_length * ground_truth_depth_left.shape[1],
            focal_length * ground_truth_depth_left.shape[1],
            0.5 * ground_truth_depth_left.shape[1], 0.5 * ground_truth_depth_left.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))
        noisy_depth = depth_noise.kinect_depth_model(ground_truth_depth_left,
                                                     ground_truth_depth_right, camera_intrinsics, relative_pose)
        self.assertLessEqual(np.max(noisy_depth), 4.1)  # A little leeway for noise
        self.assertGreaterEqual(np.min(noisy_depth[np.nonzero(noisy_depth)]), 0.7)
        self.assertGreater(np.mean(noisy_depth), 0)  # Assert that something is visible at all
        # image_utils.show_image(noisy_depth / np.max(noisy_depth), 'test depth')

    def test_kinect_noise_is_quick(self):
        ground_truth_depth_left = get_test_image('left')
        ground_truth_depth_right = get_test_image('right')

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            ground_truth_depth_left.shape[1],
            ground_truth_depth_left.shape[0],
            focal_length * ground_truth_depth_left.shape[1],
            focal_length * ground_truth_depth_left.shape[1],
            0.5 * ground_truth_depth_left.shape[1], 0.5 * ground_truth_depth_left.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))

        number = 20
        time = timeit.timeit(
            lambda: depth_noise.kinect_depth_model(ground_truth_depth_left, ground_truth_depth_right,
                                                   camera_intrinsics, relative_pose), number=number)
        # print("Noise time: {0}, total time: {1}".format(time / number, time))
        self.assertLess(time / number, 1)


def get_test_image(suffix: str):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test-depth-{}.png'.format(suffix)
    )
    if os.path.isfile(path):
        depth_image = image_utils.read_depth(path).astype(np.float16)
        # depth_image = np.asarray(depth_image, dtype=np.float32)  # Back to floats
        depth_image = np.sum(depth_image * (255, 1, 1/255, 0), axis=2)  # Rescale the channels and combine.
        # We now have depth in unreal world units, ie, centimenters. Convert to meters.
        return np.asarray(depth_image / 100, np.float16)
    else:
        raise FileNotFoundError("Could not find test image at {0}".format(path))
