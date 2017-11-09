# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import keras_frcnn.config
import systems.deep_learning.config_helpers as cfg_help


class TestConfigHelpers(unittest.TestCase):

    @mock.patch('keras_frcnn.config.K')
    def test_serialize_and_deserialize(self, mock_backend):
        mock_backend.image_dim_ordering.return_value = 'th'
        config1 = make_random_config()
        s_config1 = cfg_help.serialize_config(config1)

        config2 = cfg_help.deserialize_config(s_config1)
        s_config2 = cfg_help.serialize_config(config2)

        self.assert_config_equal(config1, config2)
        self.assertEqual(s_config1, s_config2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            config2 = cfg_help.deserialize_config(s_config2)
            s_config2 = cfg_help.serialize_config(config2)
            self.assert_config_equal(config1, config2)
            self.assertEqual(s_config1, s_config2)

    def assert_config_equal(self, cfg1, cfg2):
        self.assertEqual(cfg1.verbose, cfg2.verbose)
        self.assertEqual(cfg1.use_horizontal_flips, cfg2.use_horizontal_flips)
        self.assertEqual(cfg1.use_vertical_flips, cfg2.use_vertical_flips)
        self.assertEqual(cfg1.rot_90, cfg2.rot_90)
        self.assertEqual(cfg1.anchor_box_scales, cfg2.anchor_box_scales)
        self.assertEqual(cfg1.anchor_box_ratios, cfg2.anchor_box_ratios)
        self.assertEqual(cfg1.im_size, cfg2.im_size)
        self.assertEqual(cfg1.img_channel_mean, cfg2.img_channel_mean)
        self.assertEqual(cfg1.img_scaling_factor, cfg2.img_scaling_factor)
        self.assertEqual(cfg1.num_rois, cfg2.num_rois)
        self.assertEqual(cfg1.rpn_stride, cfg2.rpn_stride)
        self.assertEqual(cfg1.balanced_classes, cfg2.balanced_classes)
        self.assertEqual(cfg1.std_scaling, cfg2.std_scaling)
        self.assertEqual(cfg1.classifier_regr_std, cfg2.classifier_regr_std)
        self.assertEqual(cfg1.rpn_min_overlap, cfg2.rpn_min_overlap)
        self.assertEqual(cfg1.rpn_max_overlap, cfg2.rpn_max_overlap)
        self.assertEqual(cfg1.classifier_min_overlap, cfg2.classifier_min_overlap)
        self.assertEqual(cfg1.classifier_max_overlap, cfg2.classifier_max_overlap)
        self.assertEqual(cfg1.class_mapping, cfg2.class_mapping)
        self.assertEqual(cfg1.base_net_weights, cfg2.base_net_weights)
        self.assertEqual(cfg1.model_path, cfg2.model_path)


def make_random_config():
    cfg = keras_frcnn.config.Config()
    cfg.verbose = bool(np.random.randint(0, 2))
    cfg.use_horizontal_flips = bool(np.random.randint(0, 2))
    cfg.use_vertical_flips = bool(np.random.randint(0, 2))
    cfg.rot_90 = bool(np.random.randint(0, 2))
    cfg.anchor_box_scales = list(np.random.randint(0, 2048, np.random.randint(2, 5)))
    cfg.anchor_box_ratios = [[np.random.randint(1, 20), np.random.randint(1, 20)]
                             for _ in range(np.random.randint(1, 7))]
    cfg.im_size = np.random.randint(100, 2000)
    cfg.img_channel_mean = list(np.random.uniform(0, 255, 3))
    cfg.img_scaling_factor = np.random.uniform(0.5, 2)
    cfg.num_rois = np.random.randint(1, 100)
    cfg.rpn_stride = np.random.randint(1, 100)
    cfg.balanced_classes = bool(np.random.randint(0, 2))
    cfg.std_scaling = np.random.uniform(1, 10)
    cfg.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
    cfg.rpn_min_overlap = np.random.uniform(0, 1)
    cfg.rpn_max_overlap = np.random.uniform(0, 1)
    cfg.classifier_min_overlap = np.random.uniform(0, 1)
    cfg.classifier_max_overlap = np.random.uniform(0, 1)
    cfg.class_mapping = None
    cfg.base_net_weights = 'base-weights-{0}.h5'.format(np.random.randint(1000000))
    cfg.model_path = 'model_frcnn_{0}.hdf5'.format(np.random.randint(1000000))
    return cfg
