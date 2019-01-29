# Copyright 2017 John Skinner
"""
Minimal types for many of the core abstract base classes.
This allows other tests to use instances of these types.
"""
import numpy as np
import arvet.util.transform as tf
import arvet.util.dict_utils as du
import arvet.metadata.image_metadata as imeta
import arvet.core.image as im
import arvet.core.system
import arvet.core.image_source
import arvet.core.trial_result
from arvet.core.sequence_type import ImageSequenceType
import arvet.core.metric
import arvet.core.trial_comparison
import arvet.database.entity
import arvet.metadata.camera_intrinsics as cam_intr


class MockSystem(arvet.core.system.VisionSystem):
    def is_deterministic(self):
        return True

    def is_image_source_appropriate(self, image_source):
        return True

    def set_camera_intrinsics(self, camera_intrinsics):
        pass

    def start_trial(self, sequence_type):
        pass

    def process_image(self, image, timestamp):
        pass

    def finish_trial(self):
        return arvet.core.trial_result.TrialResult(
            self.identifier, True, arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})


class MockImageSource(arvet.core.image_source.ImageSource):
    sequence_type = ImageSequenceType.NON_SEQUENTIAL
    is_depth_available = False
    is_normals_available = False
    is_stereo_available = False
    is_labels_available = False
    is_masks_available = False
    is_stored_in_database = True
    camera_intrinsics = cam_intr.CameraIntrinsics()

    def __iter__(self):
        for idx in range(10):
            yield 0.6 * idx, make_image(idx)


class MockMetric(arvet.core.metric.Metric):

    def is_trial_appropriate(self, trial_result):
        return True

    def measure_results(self, trial_results):
        return arvet.core.metric.MetricResult(self, list(trial_results), True)


class MockTrialComparisonMetric(arvet.core.trial_comparison.TrialComparisonMetric):

    def is_trial_appropriate_for_first(self, trial_result):
        return True

    def is_trial_appropriate_for_second(self, trial_result):
        return True

    def compare_trials(self, trial_results_1, trial_results_2):
        return arvet.core.trial_comparison.TrialComparisonResult(
            metric=self,
            trial_results_1=list(trial_results_1),
            trial_results_2=list(trial_results_2),
            success=True
        )


def make_metadata(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'img_hash': b'\xf1\x9a\xe2|' + np.random.randint(0, 0xFFFFFFFF).to_bytes(4, 'big'),
        'source_type': imeta.ImageSourceType.SYNTHETIC,
        'camera_pose': tf.Transform(location=(1 + 100 * index, 2 + np.random.uniform(-1, 1), 3),
                                    rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
        'intrinsics': cam_intr.CameraIntrinsics(800, 600, 550.2, 750.2, 400, 300),
        'environment_type': imeta.EnvironmentType.INDOOR_CLOSE,
        'light_level': imeta.LightingLevel.WELL_LIT, 'time_of_day': imeta.TimeOfDay.DAY,
        'lens_focal_distance': 5,
        'aperture': 22,
        'simulation_world': 'TestSimulationWorld',
        'lighting_model': imeta.LightingModel.LIT,
        'texture_mipmap_bias': 1,
        'normal_maps_enabled': 2,
        'roughness_enabled': True,
        'geometry_decimation': 0.8,
        'procedural_generation_seed': 16234,
        'labelled_objects': (
            imeta.LabelledObject(
                class_names=('car',), x=12, y=144, width=67, height=43,
                relative_pose=tf.Transform((12, 3, 4), (0.5, 0.1, 1, 1.7)),
                instance_name='Car-002'
            ),
            imeta.LabelledObject(
                class_names=('cat',), x=125, y=244, width=117, height=67,
                relative_pose=tf.Transform((378, -1890, 38), (0.3, 1.12, 1.1, 0.2)),
                instance_name='cat-090'
            )
        )
    })
    return imeta.ImageMetadata(**kwargs)


def make_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'pixels': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': make_metadata(index),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth': np.random.uniform(0, 1, (32, 32)),
        'normals': np.random.uniform(0, 1, (32, 32, 3))
    })
    return im.Image(**kwargs)


def make_stereo_image(index=1, **kwargs):
    kwargs = du.defaults(kwargs, {
        'pixels': np.random.uniform(0, 255, (32, 32, 3)),
        'right_pixels': np.random.uniform(0, 255, (32, 32, 3)),
        'metadata': make_metadata(index),
        'right_metadata': make_metadata(
            index,
            camera_pose=tf.Transform(location=(1 + 100*index, 2 + np.random.uniform(-1, 1), 4),
                                     rotation=(4, 5, 6, 7 + np.random.uniform(-4, 4))),
        ),
        'additional_metadata': {
            'Source': 'Generated',
            'Resolution': {'width': 1280, 'height': 720},
            'Material Properties': {
                'BaseMipMapBias': 0,
                'RoughnessQuality': True
            }
        },
        'depth': np.random.uniform(0, 1, (32, 32)),
        'right_depth': np.random.uniform(0, 1, (32, 32)),
        'normals': np.random.uniform(0, 1, (32, 32, 3)),
        'right_normals': np.random.uniform(0, 1, (32, 32, 3))
    })
    return im.StereoImage(**kwargs)
