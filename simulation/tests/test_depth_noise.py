import os.path
import unittest
import numpy as np
import cv2
import metadata.camera_intrinsics as cam_intr
import simulation.depth_noise as depth_noise


class TestDepthNoise(unittest.TestCase):

    def test_gaussian_noise(self):
        #ground_truth_depth = make_test_image()
        #noisy_depth = depth_noise.naive_gaussian_noise(ground_truth_depth)
        #cv2.imshow('test fixed', noisy_depth / np.max(noisy_depth))
        #cv2.waitKey(0)
        pass

    def test_kinect_noise(self):
        ground_truth_depth_left, ground_truth_depth_right = get_test_images()
        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            focal_length,
            focal_length * (ground_truth_depth_left.shape[1] / ground_truth_depth_left.shape[0]),
            0.5, 0.5)
        noisy_depth = depth_noise.kinect_depth_model(ground_truth_depth_left, ground_truth_depth_right, camera_intrinsics)
        cv2.imshow('test depth linear', noisy_depth / np.max(noisy_depth))
        cv2.waitKey(0)


def get_test_images():
    return get_test_image('left'), get_test_image('right')

def get_test_image(suffix):
    if suffix is not None:
        path = os.path.expanduser('~/test-depth-{}.png'.format(suffix))
    else:
        path = os.path.expanduser('~/test-depth.png')
    if not os.path.isfile(path):
        return make_test_image()
    else:
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth_image = np.asarray(depth_image, dtype=np.float32)  # Back to floats
        depth_image = np.sum(depth_image * (1/255, 1, 255, 0), axis=2)  # Rescale the channels and combine.
        # We now have depth in unreal world units, ie, centimenters. Convert to meters.
        return depth_image / 100


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
