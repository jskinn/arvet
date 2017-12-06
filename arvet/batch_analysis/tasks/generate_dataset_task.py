# Copyright (c) 2017, John Skinner
import arvet.batch_analysis.task


class GenerateDatasetTask(arvet.batch_analysis.task.Task):
    """
    A task for generating a dataset. Result will be an image source id or list of image source ids.
    """
    def __init__(self, controller_id, simulator_id, simulator_config, repeat=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller_id = controller_id
        self._simulator_id = simulator_id
        self._simulator_config = simulator_config
        self._repeat = repeat

    @property
    def controller_id(self):
        return self._controller_id

    @property
    def simulator_id(self):
        return self._simulator_id

    @property
    def simulator_config(self):
        return self._simulator_config

    @property
    def repeat(self):
        return self._repeat

    def run_task(self, db_client):
        import logging
        import traceback
        import arvet.util.database_helpers as dh
        import arvet.image_collections.image_collection_builder as collection_builder

        # Try and import the desired loader module
        controller = dh.load_object(db_client, db_client.image_source_collection, self.controller_id)
        simulator = dh.load_object(db_client, db_client.image_source_collection, self.simulator_id,
                                   config=self.simulator_config)

        if simulator is None:
            logging.getLogger(__name__).error("Could not deserialize simulator {0}".format(self.simulator_id))
            self.mark_job_failed()
        elif controller is None:
            logging.getLogger(__name__).error("Could not deserialize controller {0}".format(self.controller_id))
            self.mark_job_failed()
        elif not controller.can_control_simulator(simulator):
            logging.getLogger(__name__).error(
                "Controller {0} can not control simulator {1}".format(self.controller_id, self.simulator_id))
            self.mark_job_failed()
        else:
            controller.set_simulator(simulator)
            logging.getLogger(__name__).info(
                "Generating dataset from {0} using controller {1}".format(self.simulator_id, self.controller_id))
            builder = collection_builder.ImageCollectionBuilder(db_client)
            try:
                builder.add_from_image_source(controller)
                dataset_id = builder.save()
            except Exception:
                dataset_id = None
                logging.getLogger(__name__).error(
                    "Exception occurred while generating dataset from simulator {0} with controller {1}:\n{2}".format(
                        self.simulator_id, self.controller_id, traceback.format_exc()
                    ))
            if dataset_id is None:
                logging.getLogger(__name__).error(
                    "Failed to generate dataset from simulator {0} with controller {1}: Dataset was null".format(
                        self.simulator_id, self.controller_id))
                self.mark_job_failed()
            else:
                self.mark_job_complete(dataset_id)
                logging.getLogger(__name__).info("Successfully generated dataset {0}".format(dataset_id))

    def serialize(self):
        serialized = super().serialize()
        serialized['controller_id'] = self.controller_id
        serialized['simulator_id'] = self.simulator_id
        serialized['simulator_config'] = self.simulator_config
        serialized['repeat'] = self._repeat
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'controller_id' in serialized_representation:
            kwargs['controller_id'] = serialized_representation['controller_id']
        if 'simulator_id' in serialized_representation:
            kwargs['simulator_id'] = serialized_representation['simulator_id']
        if 'simulator_config' in serialized_representation:
            kwargs['simulator_config'] = serialized_representation['simulator_config']
        if 'repeat' in serialized_representation:
            kwargs['repeat'] = serialized_representation['repeat']
        return super().deserialize(serialized_representation, db_client, **kwargs)
