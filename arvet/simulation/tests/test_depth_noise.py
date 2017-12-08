# Copyright (c) 2017, John Skinner
import os.path
import unittest
import numpy as np
import timeit
import arvet.util.image_utils as image_utils
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.simulation.depth_noise as depth_noise


_test_dir = 'temp-test_depth_noise'


class TestDepthNoise(unittest.TestCase):

    def test_gaussian_noise(self):
        ground_truth_depth = make_test_image()
        noisy_depth = depth_noise.naive_gaussian_noise(ground_truth_depth)

        diff = noisy_depth - ground_truth_depth
        self.assertEqual(diff.shape[0] * diff.shape[1], np.count_nonzero(diff))
        self.assertAlmostEqual(0, np.mean(diff), delta=0.1)
        self.assertAlmostEqual(0.1, np.std(diff), delta=0.05)
        # image_utils.show_image(noisy_depth / np.max(noisy_depth), 'test depth')

    def test_kinect_noise(self):
        ground_truth_depth_left, ground_truth_depth_right = get_test_images()

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            ground_truth_depth_left.shape[1],
            ground_truth_depth_left.shape[0],
            focal_length * ground_truth_depth_left.shape[1],
            focal_length * ground_truth_depth_left.shape[1],
            0.5 * ground_truth_depth_left.shape[1], 0.5 * ground_truth_depth_left.shape[1])
        noisy_depth = depth_noise.kinect_depth_model(ground_truth_depth_left,
                                                     ground_truth_depth_right, camera_intrinsics)
        self.assertLessEqual(np.max(noisy_depth), 4)
        self.assertGreaterEqual(0.8, np.min(noisy_depth))
        self.assertGreater(0, np.mean(noisy_depth)) # Assert that something is visible at all
        #image_utils.show_image(noisy_depth / np.max(noisy_depth), 'test depth')


def get_test_images():
    return get_test_image('left'), get_test_image('right')

def get_test_image(suffix: str):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test-depth-{}.png'.format(suffix)
    )
    if os.path.isfile(path):
        depth_image = image_utils.read_depth(path).astype(np.float32)
        depth_image = np.asarray(depth_image, dtype=np.float32)  # Back to floats
        depth_image = np.sum(depth_image * (255, 1, 1/255, 0), axis=2)  # Rescale the channels and combine.
        # We now have depth in unreal world units, ie, centimenters. Convert to meters.
        return depth_image / 100
    else:
        raise FileNotFoundError("Could not find test image at {0}".format(path))


def make_test_image():
    width = 1280
    height = 720
    max_depth = 10
    depth = max_depth * np.ones((height, width), dtype=np.float32)
    margin = min(height, width) // 4
    for i in range(margin):
        depth[i, i:width - i] = max_depth * i / margin
        depth[height - i - 1, i:width - i] = max_depth * i / margin
        depth[i:height - i, i] = max_depth * i / margin
        depth[i:height - i, width - i - 1] = max_depth * i / margin
    return depth
