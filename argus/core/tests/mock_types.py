# Copyright 2107 John Skinner
"""
Minimal types for many of the core abstract base classes.
This allows other tests to use instances of these types.
"""
import argus.core.system
import argus.core.image_source
import argus.core.trial_result
import argus.core.benchmark
import argus.core.sequence_type
import argus.database.entity


class MockSystem(argus.core.system.VisionSystem):
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
        return argus.core.trial_result.TrialResult(
            self.identifier, True, argus.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})


class MockImageSource(argus.core.image_source.ImageSource, argus.database.entity.Entity):
    def sequence_type(self):
        return argus.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

    @property
    def supports_random_access(self):
        return True

    @property
    def is_depth_available(self):
        return False

    @property
    def is_per_pixel_labels_available(self):
        return False

    @property
    def is_labels_available(self):
        return False

    @property
    def is_normals_available(self):
        return False

    @property
    def is_stereo_available(self):
        return False

    @property
    def is_stored_in_database(self):
        return True

    def get_camera_intrinsics(self):
        return None

    def begin(self):
        return True

    def get(self, index):
        return None

    def get_next_image(self):
        return None, None

    def is_complete(self):
        return True


class MockBenchmark(argus.core.benchmark.Benchmark):

    @classmethod
    def get_trial_requirements(cls):
        return {}

    def is_trial_appropriate(self, trial_result):
        return True

    def benchmark_results(self, trial_result):
        return argus.core.benchmark.BenchmarkResult(self.identifier, trial_result.identifier, True)
