# Copyright (c) 2017, John Skinner
import abc
import typing
import pymodm
import pymodm.fields as fields
from pathlib import PurePath
from pandas import DataFrame
import arvet.database.pymodm_abc as pymodm_abc


class Plot(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    A class that visualises results, producing plots.
    Each experiment should have some number of plots, which it will create as a task.


    This is an abstract base class defining an interface for all plots,
    to allow them to be called easily and in a structured way.
    """
    name = fields.CharField(required=True)

    @abc.abstractmethod
    def get_required_columns(self) -> typing.Set[str]:
        """
        Get the set of properties required for this plot.
        This will be passed to the get_columns of the various results to pull together a dataframe,
        which is then passed to plot_results
        :return:
        """
        pass

    @abc.abstractmethod
    def plot_results(self, data: DataFrame, output_dir: PurePath, display: bool = False) -> None:
        """
        Given the data, produce this plot.
        This should produce a file in the given folder with the same name as this plot (with some extension).
        If such a file already exists, it should overwrite it.

        :param data: The data used to create the plot. It will have at least the columns from get_required_columns
        :param output_dir: The location to save the output plot.
        :param display: Show the plots when complete
        :return:
        """
        pass

    def check_required_columns(self, data: DataFrame) -> None:
        """
        Check that the given data frame has the required columns for this plot.
        Will raise RuntimeError if any are missing, otherwise return.
        Call this at the start of plot_results.
        :param data: The DataFrame given to plot_results
        :return: None
        """
        missing = self.get_required_columns() - set(data.columns)
        if len(missing) > 0:
            raise RuntimeError(f"Data frame was missing required columns {missing}")

    @classmethod
    def get_instance(cls) -> 'Plot':
        """
        Get an instance of this plot, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        all_objects = cls.objects.all()
        if all_objects.count() > 0:
            return all_objects.first()
        obj = cls()
        return obj
