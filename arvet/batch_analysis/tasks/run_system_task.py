# Copyright (c) 2017, John Skinner
import logging
import time
import typing
import bson

import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
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

    @property
    def result_id(self) -> typing.Union[bson.ObjectId, None]:
        """
        Get the id of the result, without attempting to construct the object.
        Makes it easier for other objects to refer to this result, without loading large result objects.
        :return:
        """
        with no_auto_dereference(RunSystemTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                return self.result
        return self.result.pk

    def get_result(self) -> typing.Union[TrialResult, None]:
        """
        Actually get the result object.
        This will auto-load the result model, and then attempt to construct it.
        :return:
        """
        with no_auto_dereference(RunSystemTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                # result is an id and not a model, autoload the model
                autoload_modules(TrialResult, [self.result])
        # This will now dereference correctly
        return self.result

    def run_task(self, path_manager: PathManager):
        import traceback

        # Before we do anything, make sure we've imported the relevant models for our referenced fields.
        self.load_referenced_modules()

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

        logging.getLogger(__name__).info("Saving trial result...")
        trial_result.save()

        self.result = trial_result
        self.mark_job_complete()

    def load_referenced_modules(self):
        logging.getLogger(__name__).info("Loading referenced models...")
        # Load the system model
        model_id = None
        with no_auto_dereference(RunSystemTask):
            if isinstance(self.system, bson.ObjectId):
                model_id = self.system
        if model_id is not None:
            autoload_modules(VisionSystem, [model_id])

        # Load the image source model
        model_id = None
        with no_auto_dereference(RunSystemTask):
            if isinstance(self.image_source, bson.ObjectId):
                model_id = self.image_source
        if model_id is not None:
            autoload_modules(ImageSource, [model_id])

    def fail_with_message(self, message):
        """
        Quick helper to log error message, and make and store a trial result as the result
        :param message:
        :return:
        """
        logging.getLogger(__name__).error(message)
        self.result = TrialResult(
            system=self.system,
            image_source=self.image_source,
            message=message
        )
        self.mark_job_complete()


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
    if image_source.stereo_offset is not None:
        system.set_stereo_offset(image_source.stereo_offset)
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
