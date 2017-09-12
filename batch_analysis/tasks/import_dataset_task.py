#Copyright (c) 2017, John Skinner
import batch_analysis.task


class ImportDatasetTask(batch_analysis.task.Task):
    """
    A task for importing a dataset. Result will be an image source id or list of image source ids.
    """
    def __init__(self, module_name, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_name = module_name
        self._path = path

    @property
    def module_name(self):
        return self._module_name

    @property
    def path(self):
        return self._path

    def run_task(self, db_client):
        import logging
        import traceback
        import importlib

        # Try and import the desired loader module
        try:
            loader_module = importlib.import_module(self.module_name)
        except ImportError:
            loader_module = None

        if loader_module is None:
            logging.getLogger(__name__).error("Could not load module {0} for importing dataset, check it  exists".format(self.module_name))
            self.mark_job_failed()
        elif not hasattr(loader_module, 'import_dataset'):
            logging.getLogger(__name__).error("Module {0} does not have method 'import_dataset'".format(self.module_name))
            self.mark_job_failed()
        else:
            logging.getLogger(__name__).info("Importing dataset from {0} using module {1}".format(self.path, self.module_name))
            try:
                dataset_id = loader_module.import_dataset(self.path, db_client)
            except Exception:
                dataset_id = None
                logging.getLogger(__name__).error("Exception occurred while importing dataset from {0} with module {1}:\n{2}".format(
                    self.path, self.module_name, traceback.format_exc()
                ))

            if dataset_id is None:
                logging.getLogger(__name__).error("Failed to import dataset from {0} with module {1}".format(
                    self.path, self.module_name))
                self.mark_job_failed()
            else:
                self.mark_job_complete(dataset_id)
                logging.getLogger(__name__).info("Successfully imported dataset {0}".format(dataset_id))

    def serialize(self):
        serialized = super().serialize()
        serialized['module_name'] = self.module_name
        serialized['path'] = self.path
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'module_name' in serialized_representation:
            kwargs['module_name'] = serialized_representation['module_name']
        if 'path' in serialized_representation:
            kwargs['path'] = serialized_representation['path']
        return super().deserialize(serialized_representation, db_client, **kwargs)
