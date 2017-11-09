# Copyright (c) 2017, John Skinner
"""
Helpers for serializing and deserializing keras_frcnn config to the database
"""
import keras_frcnn.config


def serialize_config(config):
    return {
        'verbose': config.verbose,
        'use_horizontal_flips': config.use_horizontal_flips,
        'use_vertical_flips': config.use_vertical_flips,
        'rot_90': config.rot_90,
        'anchor_box_scales': config.anchor_box_scales,
        'anchor_box_ratios': config.anchor_box_ratios,
        'im_size': config.im_size,
        'img_channel_mean': config.img_channel_mean,
        'img_scaling_factor': config.img_scaling_factor,
        'num_rois': config.num_rois,
        'rpn_stride': config.rpn_stride,
        'balanced_classes': config.balanced_classes,
        'std_scaling': config.std_scaling,
        'classifier_regr_std': config.classifier_regr_std,
        'rpn_min_overlap': config.rpn_min_overlap,
        'rpn_max_overlap': config.rpn_max_overlap,
        'classifier_min_overlap': config.classifier_min_overlap,
        'classifier_max_overlap': config.classifier_max_overlap,
        'class_mapping': config.class_mapping,
        'base_net_weights': config.base_net_weights,
        'model_path': config.model_path
    }


def deserialize_config(s_config):
    config = keras_frcnn.config.Config()
    if 'verbose' in s_config:
        config.verbose = s_config['verbose']
    if 'use_horizontal_flips' in s_config:
        config.use_horizontal_flips = s_config['use_horizontal_flips']
    if 'use_vertical_flips' in s_config:
        config.use_vertical_flips = s_config['use_vertical_flips']
    if 'rot_90' in s_config:
        config.rot_90 = s_config['rot_90']
    if 'anchor_box_scales' in s_config:
        config.anchor_box_scales = s_config['anchor_box_scales']
    if 'anchor_box_ratios' in s_config:
        config.anchor_box_ratios = s_config['anchor_box_ratios']
    if 'im_size' in s_config:
        config.im_size = s_config['im_size']
    if 'img_channel_mean' in s_config:
        config.img_channel_mean = s_config['img_channel_mean']
    if 'img_scaling_factor' in s_config:
        config.img_scaling_factor = s_config['img_scaling_factor']
    if 'num_rois' in s_config:
        config.num_rois = s_config['num_rois']
    if 'rpn_stride' in s_config:
        config.rpn_stride = s_config['rpn_stride']
    if 'balanced_classes' in s_config:
        config.balanced_classes = s_config['balanced_classes']
    if 'std_scaling' in s_config:
        config.std_scaling = s_config['std_scaling']
    if 'classifier_regr_std' in s_config:
        config.classifier_regr_std = s_config['classifier_regr_std']
    if 'rpn_min_overlap' in s_config:
        config.rpn_min_overlap = s_config['rpn_min_overlap']
    if 'rpn_max_overlap' in s_config:
        config.rpn_max_overlap = s_config['rpn_max_overlap']
    if 'classifier_min_overlap' in s_config:
        config.classifier_min_overlap = s_config['classifier_min_overlap']
    if 'classifier_max_overlap' in s_config:
        config.classifier_max_overlap = s_config['classifier_max_overlap']
    if 'class_mapping' in s_config:
        config.class_mapping = s_config['class_mapping']
    if 'base_net_weights' in s_config:
        config.base_net_weights = s_config['base_net_weights']
    if 'model_path' in s_config:
        config.model_path = s_config['model_path']
    return config
