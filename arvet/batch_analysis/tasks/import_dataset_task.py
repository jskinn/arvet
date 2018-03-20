# Copyright (c) 2017, John Skinner
import arvet.batch_analysis.task
import arvet.database.client
import arvet.config.path_manager


class ImportDatasetTask(arvet.batch_analysis.task.Task):
    """
    A task for importing a dataset. Result will be an image source id
    """
    def __init__(self, module_name, path, additional_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_name = module_name
        self._path = path
        self._additional_args = additional_args if additional_args is not None else {}

    @property
    def module_name(self):
        return self._module_name

    @property
    def path(self):
        return self._path

    @property
    def additional_args(self):
        return self._additional_args

    def run_task(self, path_manager: arvet.config.path_manager.PathManager,
                 db_client: arvet.database.client.DatabaseClient):
        import logging
        import traceback
        import importlib

        # Try and import the desired loader module
        try:
            loader_module = importlib.import_module(self.module_name)
        except ImportError:
            loader_module = None

        # Try and find the root directory or file to load the dataset from
        try:
            actual_path = path_manager.find_path(self.path)
        except FileNotFoundError:
            logging.getLogger(__name__).error(
                "Could not find dataset path {0}".format(self.path))
            self.mark_job_failed()
            actual_path = None

        if loader_module is None:
            logging.getLogger(__name__).error(
                "Could not load module {0} for importing dataset, check it  exists".format(self.module_name))
            self.mark_job_failed()
        elif not hasattr(loader_module, 'import_dataset'):
            logging.getLogger(__name__).error(
                "Module {0} does not have method 'import_dataset'".format(self.module_name))
            self.mark_job_failed()
        elif actual_path is not None:
            logging.getLogger(__name__).info(
                "Importing dataset from {0} using module {1}".format(actual_path, self.module_name))
            # noinspection PyBroadException
            try:
                dataset_id = loader_module.import_dataset(actual_path, db_client, **self.additional_args)
            except Exception as exception:
                logging.getLogger(__name__).error(
                    "Exception occurred while importing dataset from {0} with module {1}:\n{2}".format(
                        actual_path, self.module_name, traceback.format_exc()
                    ))
                self.mark_job_failed()
                raise exception

            if dataset_id is None:
                logging.getLogger(__name__).error("Failed to import dataset from {0} with module {1}".format(
                    actual_path, self.module_name))
                self.mark_job_failed()
            else:
                self.mark_job_complete(dataset_id)
                logging.getLogger(__name__).info("Successfully imported dataset {0}".format(dataset_id))

    def serialize(self):
        serialized = super().serialize()
        serialized['module_name'] = self.module_name
        serialized['path'] = self.path
        serialized['additional_args'] = self.additional_args
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'module_name' in serialized_representation:
            kwargs['module_name'] = serialized_representation['module_name']
        if 'path' in serialized_representation:
            kwargs['path'] = serialized_representation['path']
        if 'additional_args' in serialized_representation:
            kwargs['additional_args'] = serialized_representation['additional_args']
        return super().deserialize(serialized_representation, db_client, **kwargs)
