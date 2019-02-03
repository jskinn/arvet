# Copyright (c) 2017, John Skinner
import pymodm.fields as fields
from arvet.core.image_collection import ImageCollection
from arvet.batch_analysis.task import Task
from arvet.config.path_manager import PathManager


class ImportDatasetTask(Task):
    """
    A task for importing a dataset. Result will be an image source id
    """
    module_name = fields.CharField(required=True)
    path = fields.CharField(required=True)
    additional_args = fields.DictField(default={}, blank=True)
    result = fields.ReferenceField(ImageCollection)

    def run_task(self, path_manager: PathManager):
        import logging
        import traceback
        import importlib

        # Try and import the desired loader module
        try:
            loader_module = importlib.import_module(self.module_name)
        except ImportError as exception:
            logging.getLogger(__name__).error(
                "Could not load module {0} for importing dataset, check it  exists".format(self.module_name))
            self.mark_job_failed()
            raise exception

        # Check the module has the required function
        if not hasattr(loader_module, 'import_dataset'):
            logging.getLogger(__name__).error(
                "Module {0} does not have method 'import_dataset'".format(self.module_name))
            self.mark_job_failed()
            return

        # Try and find the root directory or file to load the dataset from
        try:
            actual_path = path_manager.find_path(self.path)
        except FileNotFoundError:
            logging.getLogger(__name__).error(
                "Could not find dataset path {0}".format(self.path))
            self.mark_job_failed()
            return

        logging.getLogger(__name__).info(
            "Importing dataset from {0} using module {1}".format(actual_path, self.module_name))
        try:
            image_collection = loader_module.import_dataset(actual_path, **self.additional_args)
        except Exception as exception:
            logging.getLogger(__name__).error(
                "Exception occurred while importing dataset from {0} with module {1}:\n{2}".format(
                    actual_path, self.module_name, traceback.format_exc()
                ))
            self.mark_job_failed()
            raise exception

        if image_collection is None:
            logging.getLogger(__name__).error("Failed to import dataset from {0} with module {1}".format(
                actual_path, self.module_name))
            self.mark_job_failed()
        else:
            self.result = image_collection
            self.mark_job_complete()
            logging.getLogger(__name__).info("Successfully imported dataset")
