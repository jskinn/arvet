# Copyright (c) 2017, John Skinner
import arvet.batch_analysis.task
import arvet.database.client
import arvet.database.entity_registry as entity_registry
import arvet.config.path_manager


class RunSystemTask(arvet.batch_analysis.task.Task):
    """
    A task for running a system with an image source. Result will be a trial result id.
    """
    def __init__(self, system_id, image_source_id, repeat=0, *args, **kwargs):
        """
        Create a run system task
        :param system_id: The system to run
        :param image_source_id: The image
        :param repeat: The number repetition of this combination, so that we can run the same combination more than once
        :param args: Args passed to the Task constructor
        :param kwargs: Kwargs passed to the Task constructor
        """
        super().__init__(*args, **kwargs)
        self._system = system_id
        self._image_source = image_source_id
        self._repeat = repeat

    @property
    def system(self):
        return self._system

    @property
    def image_source(self):
        return self._image_source

    @property
    def repeat(self):
        return self._repeat

    def run_task(self, path_manager: arvet.config.path_manager.PathManager,
                 db_client: arvet.database.client.DatabaseClient):
        import logging
        import traceback
        import arvet.util.database_helpers as dbhelp

        system = dbhelp.load_object(db_client, db_client.system_collection, self.system)
        image_source = dbhelp.load_object(db_client, db_client.image_source_collection, self.image_source)

        if system is None:
            logging.getLogger(__name__).error("Could not deserialize system {0}".format(self.system))
            self.mark_job_failed()
        elif image_source is None:
            logging.getLogger(__name__).error("Could not deserialize image source {0}".format(self.image_source))
            self.mark_job_failed()
        elif not system.is_image_source_appropriate(image_source):
            logging.getLogger(__name__).error("Image source {0} is inappropriate for system {1}".format(
                self.image_source, self.system))
            self.mark_job_failed()
        else:
            system.resolve_paths(path_manager)
            logging.getLogger(__name__).info("Start running system {0} ({1}) with image source {2}".format(
                self.system,
                entity_registry.get_type_name(type(system)),
                self.image_source))
            try:
                trial_result = run_system_with_source(system, image_source)
            except Exception:
                logging.getLogger(__name__).error("Error occurred while running system {0} "
                                                  "with image source {1}:\n{2}".format(
                    self.system, self.image_source, traceback.format_exc()))
                trial_result = None
            if trial_result is None:
                logging.getLogger(__name__).error("Failed to system {0} with image source {1}.".format(
                    self.system, self.image_source))
                self.mark_job_failed()
            else:
                trial_result.save_data(db_client)
                trial_result_id = db_client.trials_collection.insert(trial_result.serialize())
                logging.getLogger(__name__).info(("Successfully ran system {0} with image source {1},"
                                                  "producing trial result {2}").format(
                    self.system, self.image_source, trial_result_id))
                self.mark_job_complete(trial_result_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['system_id'] = self.system
        serialized['image_source_id'] = self.image_source
        serialized['repeat'] = self._repeat
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'system_id' in serialized_representation:
            kwargs['system_id'] = serialized_representation['system_id']
        if 'image_source_id' in serialized_representation:
            kwargs['image_source_id'] = serialized_representation['image_source_id']
        if 'repeat' in serialized_representation:
            kwargs['repeat'] = serialized_representation['repeat']
        return super().deserialize(serialized_representation, db_client, **kwargs)


def run_system_with_source(system, image_source):
    """
    Run a given vision system with a given image source.
    This is the structure for how image sources and vision systems should be interacted with.
    Both should already be set up and configured.
    :param system: The system to run.
    :param image_source: The image source to get images from
    :return: The TrialResult storing the results of the run. Save it to the database, or None if there's a problem.
    """
    if system.is_image_source_appropriate(image_source):
        with image_source:
            system.set_camera_intrinsics(image_source.get_camera_intrinsics())
            stereo_baseline = image_source.get_stereo_baseline()
            if stereo_baseline is not None:
                system.set_stereo_baseline(stereo_baseline)

            system.start_trial(image_source.sequence_type)
            while not image_source.is_complete():
                image, timestamp = image_source.get_next_image()
                system.process_image(image, timestamp)
        return system.finish_trial()
    return None
