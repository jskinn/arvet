# Copyright (c) 2017, John Skinner
import arvet.batch_analysis.task
import arvet.database.client
import arvet.database.entity_registry as entity_registry
import arvet.config.path_manager


class TrainSystemTask(arvet.batch_analysis.task.Task):
    """
    A task for training a system. Result will be a system id
    """
    def __init__(self, trainer_id, trainee_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainer = trainer_id
        self._trainee = trainee_id

    @property
    def trainer(self):
        return self._trainer

    @property
    def trainee(self):
        return self._trainee

    def run_task(self, path_manager: arvet.config.path_manager.PathManager,
                 db_client: arvet.database.client.DatabaseClient):
        import logging
        import traceback
        import arvet.util.database_helpers as dh

        trainer = dh.load_object(db_client, db_client.trainer_collection, self.trainer)
        trainee = dh.load_object(db_client, db_client.trainee_collection, self.trainee)

        if trainer is None:
            logging.getLogger(__name__).error("Could not deserialize trainer {0}".format(self.trainer))
            self.mark_job_failed()
        elif trainee is None:
            logging.getLogger(__name__).error("Could not deserialize trainee {0}".format(self.trainee))
            self.mark_job_failed()
        elif not trainer.can_train_trainee(trainee):
            logging.getLogger(__name__).error("Trainer {0} cannot train trainee {1}".format(
                self.trainer, self.trainee))
            self.mark_job_failed()
        else:
            logging.getLogger(__name__).info("Start training trainee {0} ({1}) with trainer {2} (3)".format(
                self.trainee,
                entity_registry.get_type_name(type(trainee)),
                self.trainer,
                entity_registry.get_type_name(type(trainer))
            ))
            try:
                system = trainer.train_vision_system(trainee)
            except Exception as exception:
                logging.getLogger(__name__).error("Error occurred while trainer {0} trains trainee {1}:\n{2}".format(
                    self.trainer,
                    self.trainee,
                    traceback.format_exc()
                ))
                self.mark_job_failed()
                raise exception
            if system is None:
                logging.getLogger(__name__).error("Failed to train trainee {0} with trainer {1}".format(
                    self.trainer, self.trainee))
                self.mark_job_failed()
            else:
                system_id = db_client.system_collection.insert(system.serialize())
                logging.getLogger(__name__).info("Successfully trained system {0}".format(system_id))
                self.mark_job_complete(system_id)

    def serialize(self):
        serialized = super().serialize()
        serialized['trainer_id'] = self.trainer
        serialized['trainee_id'] = self.trainee
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'trainer_id' in serialized_representation:
            kwargs['trainer_id'] = serialized_representation['trainer_id']
        if 'trainee_id' in serialized_representation:
            kwargs['trainee_id'] = serialized_representation['trainee_id']
        return super().deserialize(serialized_representation, db_client, **kwargs)
