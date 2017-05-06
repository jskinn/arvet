import unittest
import numpy as np
import util.transform as tf
import util.dict_utils as du
import core.image as im


class TestImage(unittest.TestCase):

    def setUp(self):
        trans = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        self.image = im.Image(
            timestamp=13/30,
            filename='/home/user/test.png',
            camera_pose=trans)

        trans = tf.Transform((4, 5, 6), (0.5, -0.5, 0.5, -0.5))
        self.full_image = im.Image(
            timestamp=14/31,
            filename='/home/user/test2.png',
            camera_pose=trans,
            depth_filename='/home/user/test2_depth.png',
            labels_filename='/home/user/test2_labels.png',
            world_normals_filename='/home/user/test2_normals.png',
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        )

    def test_filename(self):
        self.assertEqual(self.image.filename, '/home/user/test.png')
        self.assertEqual(self.full_image.filename, '/home/user/test2.png')

    def test_timestamp(self):
        self.assertEqual(self.image.timestamp, 13/30)
        self.assertEqual(self.full_image.timestamp, 14/31)

    def test_camera_location(self):
        self.assertTrue(np.array_equal(self.image.camera_location, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(self.full_image.camera_location, np.array([4, 5, 6])))

    def test_camera_orientation(self):
        self.assertTrue(np.array_equal(self.image.camera_orientation, np.array([0.5, 0.5, -0.5, -0.5])))
        self.assertTrue(np.array_equal(self.full_image.camera_orientation, np.array([0.5, -0.5, 0.5, -0.5])))

    def test_depth_filename(self):
        self.assertEqual(self.image.depth_filename, None)
        self.assertEqual(self.full_image.depth_filename, '/home/user/test2_depth.png')

    def test_labels_filename(self):
        self.assertEqual(self.image.labels_filename, None)
        self.assertEqual(self.full_image.labels_filename, '/home/user/test2_labels.png')

    def test_world_normals_filename(self):
        self.assertEqual(self.image.world_normals_filename, None)
        self.assertEqual(self.full_image.world_normals_filename, '/home/user/test2_normals.png')

    def test_additional_metadata(self):
        self.assertEqual(self.image.additional_metadata, {})
        self.assertEqual(self.full_image.additional_metadata, {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        })

    def test_padded_kwargs(self):
        kwargs = {'a': 1, 'b': 2, 'c': 3}
        with self.assertRaises(TypeError):
            im.Image(13/30, '/home/user/test.png', np.array([1, 2, 3]), np.array([1, 2, 3, 4]), **kwargs)


class TestStereoImage(unittest.TestCase):

    def setUp(self):
        self.left_pose = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        self.right_pose = tf.Transform(location=self.left_pose.find_independent((0, 0, 15)),
                                  rotation=self.left_pose.rotation_quat(w_first=False),
                                  w_first=False)
        self.image = im.StereoImage(timestamp=13/30,
                                    left_filename='/home/user/left.png',
                                    left_camera_pose=self.left_pose,
                                    right_filename='/home/user/right.png',
                                    right_camera_pose=self.right_pose)
        self.full_image = im.StereoImage(
            timestamp=31 / 67,
            left_filename='/home/user/left2.png',
            right_filename='/home/user/right2.png',
            left_camera_pose=self.left_pose,
            right_camera_pose=self.right_pose,
            left_depth_filename='/home/user/left2_depth.png',
            right_depth_filename='/home/user/right2_depth.png',
            left_labels_filename='/home/user/left2_labels.png',
            right_labels_filename='/home/user/right2_labels.png',
            left_world_normals_filename='/home/user/left2_normals.png',
            right_world_normals_filename='/home/user/right2_normals.png',
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        )

    def test_filename(self):
        self.assertEqual(self.image.left_filename, '/home/user/left.png')
        self.assertEqual(self.image.right_filename, '/home/user/right.png')
        self.assertEqual(self.full_image.left_filename, '/home/user/left2.png')
        self.assertEqual(self.full_image.right_filename, '/home/user/right2.png')

    def test_timestamp(self):
        self.assertEqual(self.image.timestamp, 13/30)
        self.assertEqual(self.full_image.timestamp, 31/67)

    def test_camera_location(self):
        self.assertTrue(np.array_equal(self.image.left_camera_location, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(self.image.right_camera_location, self.right_pose.location))
        self.assertTrue(np.array_equal(self.full_image.left_camera_location, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(self.full_image.right_camera_location, self.right_pose.location))

    def test_camera_orientation(self):
        self.assertTrue(np.array_equal(self.image.left_camera_orientation, np.array([0.5, 0.5, -0.5, -0.5])))
        self.assertTrue(np.array_equal(self.image.right_camera_orientation,
                                       self.right_pose.rotation_quat(w_first=False)))
        self.assertTrue(np.array_equal(self.full_image.left_camera_orientation, np.array([0.5, 0.5, -0.5, -0.5])))
        self.assertTrue(np.array_equal(self.full_image.right_camera_orientation,
                                       self.right_pose.rotation_quat(w_first=False)))

    def test_depth_filename(self):
        self.assertEqual(self.image.left_depth_filename, None)
        self.assertEqual(self.image.right_depth_filename, None)
        self.assertEqual(self.full_image.left_depth_filename, '/home/user/left2_depth.png')
        self.assertEqual(self.full_image.right_depth_filename, '/home/user/right2_depth.png')

    def test_labels_filename(self):
        self.assertEqual(self.image.left_labels_filename, None)
        self.assertEqual(self.image.right_labels_filename, None)
        self.assertEqual(self.full_image.left_labels_filename, '/home/user/left2_labels.png')
        self.assertEqual(self.full_image.right_labels_filename, '/home/user/right2_labels.png')

    def test_world_normals_filename(self):
        self.assertEqual(self.image.left_world_normals_filename, None)
        self.assertEqual(self.image.right_world_normals_filename, None)
        self.assertEqual(self.full_image.left_world_normals_filename, '/home/user/left2_normals.png')
        self.assertEqual(self.full_image.right_world_normals_filename, '/home/user/right2_normals.png')

    def test_additional_metadata(self):
        self.assertEqual(self.image.additional_metadata, {})
        self.assertEqual(self.full_image.additional_metadata, {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        })

    def test_padded_kwargs(self):
        kwargs = {
            'timestamp': 13 / 30,
            'left_filename': '/home/user/left.png',
            'left_camera_location': np.array([1, 2, 3]),
            'left_camera_orientation': np.array([4, 5, 6, 7]),
            'right_filename': '/home/user/right.png',
            'right_camera_location': np.array([8, 9, 10]),
            'right_camera_orientation': np.array([11, 12, 13, 14]),
            # Extras:
            'a': 1, 'b': 2, 'c': 3
        }
        with self.assertRaises(TypeError):
            im.StereoImage(**kwargs)

    def test_left_image_is_base(self):
        self.assertEqual(self.image.filename, self.image.left_filename)
        self.assertTrue(np.array_equal(self.image.camera_location, self.image.left_camera_location))
        self.assertTrue(np.array_equal(self.image.camera_orientation, self.image.left_camera_orientation))
        self.assertEqual(self.image.filename, self.image.left_filename)
        self.assertEqual(self.image.depth_filename, self.image.left_depth_filename)
        self.assertEqual(self.image.labels_filename, self.image.left_labels_filename)
        self.assertEqual(self.image.world_normals_filename, self.image.left_world_normals_filename)

    def test_make_from_images(self):
        left_pose = tf.Transform((1, 2, 3), (0.5, 0.5, -0.5, -0.5))
        left_image = im.Image(
            timestamp=14/31,
            filename='/home/user/left.png',
            camera_pose=left_pose,
            depth_filename='/home/user/left_depth.png',
            labels_filename='/home/user/left_labels.png',
            world_normals_filename='/home/user/left_normals.png',
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 0,
                    'RoughnessQuality': True
                }
            }
        )

        right_pose = tf.Transform(location=left_pose.find_independent((0, 0, 15)),
                                  rotation=left_pose.rotation_quat(w_first=False),
                                  w_first=False)
        right_image = im.Image(
            timestamp=13/31,
            filename='/home/user/right.png',
            camera_pose=right_pose,
            depth_filename='/home/user/right_depth.png',
            labels_filename='/home/user/right_labels.png',
            world_normals_filename='/home/user/right_normals.png',
            additional_metadata={
                'Source': 'Generated',
                'Resolution': {'width': 1280, 'height': 720},
                'Material Properties': {
                    'BaseMipMapBias': 1,
                    'RoughnessQuality': False
                },
                'skeletons': 'There is already one inside you'
            }
        )

        stereo_image = im.StereoImage.make_from_images(left_image, right_image)
        self.assertEqual(stereo_image.timestamp, left_image.timestamp)
        self.assertEqual(stereo_image.additional_metadata,
                         du.defaults(left_image.additional_metadata, right_image.additional_metadata))
        self.assertEqual(stereo_image.left_camera_pose, left_image.camera_pose)
        self.assertEqual(stereo_image.left_filename, left_image.filename)
        self.assertEqual(stereo_image.left_depth_filename, left_image.depth_filename)
        self.assertEqual(stereo_image.left_labels_filename, left_image.labels_filename)
        self.assertEqual(stereo_image.left_world_normals_filename, left_image.world_normals_filename)

        self.assertEqual(stereo_image.right_camera_pose, right_image.camera_pose)
        self.assertEqual(stereo_image.right_filename, right_image.filename)
        self.assertEqual(stereo_image.right_depth_filename, right_image.depth_filename)
        self.assertEqual(stereo_image.right_labels_filename, right_image.labels_filename)
        self.assertEqual(stereo_image.right_world_normals_filename, right_image.world_normals_filename)
