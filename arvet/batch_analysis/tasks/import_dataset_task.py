# Copyright (c) 2017, John Skinner
import typing
import bson
import pymodm.fields as fields
from pymodm.context_managers import no_auto_dereference
from arvet.database.autoload_modules import autoload_modules
from arvet.core.image_source import ImageSource
from arvet.batch_analysis.task import Task
from arvet.config.path_manager import PathManager


class ImportDatasetTask(Task):
    """
    A task for importing a dataset. Result will be an image source id
    """
    module_name = fields.CharField(required=True)
    path = fields.CharField(required=True)
    additional_args = fields.DictField(default={}, blank=True)
    result = fields.ReferenceField(ImageSource, on_delete=fields.ReferenceField.CASCADE)

    def get_unique_name(self) -> str:
        if 'dataset_name' in self.additional_args:
            name = self.additional_args['dataset_name'].replace(' ', '_')
            return 'import_{0}_{1}'.format(name, self.pk)
        return "import_{0}".format(self.pk)

    def load_referenced_models(self) -> None:
        """
        Load the result type so we can save the task
        :return:
        """
        with no_auto_dereference(ImportDatasetTask):
            if isinstance(self.result, bson.ObjectId):
                # result is an id and not a model, autoload the model
                autoload_modules(ImageSource, [self.result])

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

    @property
    def result_id(self) -> typing.Union[bson.ObjectId, None]:
        """
        Get the id of the result, without attempting to construct the object.
        Makes it easier for other objects to refer to this result, without loading large result objects.
        :return:
        """
        with no_auto_dereference(ImportDatasetTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                return self.result
        return self.result.pk

    def get_result(self) -> typing.Union[ImageSource, None]:
        """
        Actually get the result object.
        This will auto-load the result model, and then attempt to construct it.
        :return:
        """
        with no_auto_dereference(ImportDatasetTask):
            if self.result is None:
                return None
            if isinstance(self.result, bson.ObjectId):
                # result is an id and not a model, autoload the model
                autoload_modules(ImageSource, [self.result])
        # This will now dereference correctly
        return self.result
