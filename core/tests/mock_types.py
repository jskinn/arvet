# Copyright 2107 John Skinner
"""
Minimal types for many of the core abstract base classes.
This allows other tests to use instances of these types.
"""
import core.system
import core.image_source
import core.trial_result
import core.benchmark
import core.sequence_type
import database.entity


class MockSystem(core.system.VisionSystem):
    def is_deterministic(self):
        return True

    def is_image_source_appropriate(self, image_source):
        return True

    def set_camera_intrinsics(self, camera_intrinsics, resolution):
        pass

    def start_trial(self, sequence_type):
        pass

    def process_image(self, image, timestamp):
        pass

    def finish_trial(self):
        return core.trial_result.TrialResult(
            self.identifier, True, core.sequence_type.ImageSequenceType.NON_SEQUENTIAL, {})


class MockImageSource(core.image_source.ImageSource, database.entity.Entity):
    def sequence_type(self):
        return core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

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


class MockBenchmark(core.benchmark.Benchmark):

    @classmethod
    def get_trial_requirements(cls):
        return {}

    def is_trial_appropriate(self, trial_result):
        return True

    def benchmark_results(self, trial_result):
        return core.benchmark.BenchmarkResult(self.identifier, trial_result.identifier, True)
