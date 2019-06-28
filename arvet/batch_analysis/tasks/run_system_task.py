# Copyright (c) 2017, John Skinner
import logging
import time
import typing

import pymodm.fields as fields
from arvet.config.path_manager import PathManager
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.trial_result import TrialResult
import arvet.batch_analysis.task


class RunSystemTask(arvet.batch_analysis.task.Task):
    """
    A task for running a system with an image source. Result will be a trial result id.
    """
    system = fields.ReferenceField(VisionSystem, required=True, on_delete=fields.ReferenceField.CASCADE)
    image_source = fields.ReferenceField(ImageSource, required=True, on_delete=fields.ReferenceField.CASCADE)
    repeat = fields.IntegerField(default=0, required=True)
    result = fields.ReferenceField(TrialResult, on_delete=fields.ReferenceField.CASCADE)

    def run_task(self, path_manager: PathManager):
        import traceback

        if not self.system.is_image_source_appropriate(self.image_source):
            self.fail_with_message("Image source is inappropriate for system {0}".format(self.system.get_pretty_name()))
            return

        self.system.resolve_paths(path_manager)
        logging.getLogger(__name__).info("Start running system {0}...".format(self.system.get_pretty_name()))
        try:
            trial_result = run_system_with_source(self.system, self.image_source)
        except Exception as exception:
            self.fail_with_message("Error occurred while running system {0}:\n{1}".format(
                    self.system.get_pretty_name(), traceback.format_exc()))
            raise exception

        if not trial_result.success:
            logging.getLogger(__name__).info(
                "Ran system {0}, but got unsuccessful result: {1}".format(
                    self.system.get_pretty_name(), trial_result.message))
        else:
            logging.getLogger(__name__).info("Successfully ran system {0}".format(self.system.get_pretty_name()))
        self.result = trial_result
        self.mark_job_complete()

    def fail_with_message(self, message):
        """
        Quick helper to log error message, and make and store a trial result as the result
        :param message:
        :return:
        """
        logging.getLogger(__name__).error(message)
        self.result = FailedTaskTrialResult(
            system=self.system,
            image_source=self.image_source,
            message=message
        )
        self.mark_job_complete()


class FailedTaskTrialResult(TrialResult):
    """
    A tiny Trial Result that records a failure. Has no columns, because
    """
    success = False

    def get_columns(self) -> typing.Set[str]:
        return set()

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        return {}


def run_system_with_source(system: VisionSystem, image_source: ImageSource) -> TrialResult:
    """
    Run a given vision system with a given image source.
    This is the structure for how image sources and vision systems should be interacted with.
    Both should already be set up and configured.
    :param system: The system to run.
    :param image_source: The image source to get images from
    :return: The TrialResult storing the results of the run. Save it to the database, or None if there's a problem.
    """
    system.set_camera_intrinsics(image_source.camera_intrinsics)
    if image_source.right_camera_pose is not None:
        system.set_stereo_offset(image_source.right_camera_pose)
    logging.getLogger(__name__).info("  Initialized system")

    # Preload images into memory, so we don't have to wait while the system is running
    for _, image in image_source:
        system.preload_image_data(image)
    logging.getLogger(__name__).info("  Pre-loaded images")

    # Actually run the system, tracking the time between frames
    previous_timestamp = None
    previous_actual = None
    start_time = time.time()
    system.start_trial(image_source.sequence_type)
    for timestamp, image in image_source:
        system.process_image(image, timestamp)
        actual_time = time.time()

        if previous_timestamp is not None and previous_actual is not None:
            stamp_diff = timestamp - previous_timestamp
            actual_diff = actual_time - previous_actual
            if actual_diff > stamp_diff:
                logging.getLogger(__name__).warning("  Frame delta-time {0} exceeded timestamp delta {1}".format(
                    actual_diff, stamp_diff))

        previous_timestamp = timestamp
        previous_actual = actual_time

    trial_result = system.finish_trial()
    finish_time = time.time()
    logging.getLogger(__name__).info("  Finished running system in {0}s".format(finish_time - start_time))

    trial_result.image_source = image_source
    trial_result.duration = finish_time - start_time
    return trial_result
