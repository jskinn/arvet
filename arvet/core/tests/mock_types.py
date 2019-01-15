# Copyright 2017 John Skinner
"""
Minimal types for many of the core abstract base classes.
This allows other tests to use instances of these types.
"""
import arvet.core.system
import arvet.core.image_source
import arvet.core.trial_result
from arvet.core.sequence_type import ImageSequenceType
import arvet.core.metric
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


class MockMetric(arvet.core.metric.Metric):

    @classmethod
    def get_trial_requirements(cls):
        return {}

    def is_trial_appropriate(self, trial_result):
        return True

    def benchmark_results(self, trial_results):
        return arvet.core.metric.MetricResult(self, list(trial_results), True)
