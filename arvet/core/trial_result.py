# Copyright (c) 2017, John Skinner
import abc
import typing
import pymodm
import bson
import arvet.database.pymodm_abc as pymodm_abc
import arvet.core.system as system
import arvet.core.image_source as image_source


class TrialResult(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    """
    The result of running a particular system with images from a particular image source.
    Contains all the relevant information from the run, and is passed to the benchmark to measure the performance.
    THIS MUST INCLUDE THE GROUND-TRUTH, for whatever the system is trying to measure.
    Different subtypes of VisionSystem will have different subclasses of this.

    All Trial results have a one to many relationship with a particular dataset and system.

    Attributes:
        system          The ID of the system which produced this result
        image_source    The image source used for this trial
        success         Did the run succeed or not?
        sequence_type   The type of image sequence used to produce this result
        settings        The settings used by the system when it produced this result
    """

    system = pymodm.fields.ReferenceField(system.VisionSystem, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    image_source = pymodm.fields.ReferenceField(image_source.ImageSource, required=True,
                                                on_delete=pymodm.ReferenceField.CASCADE)
    success = pymodm.fields.BooleanField(required=True)
    run_time = pymodm.FloatField()
    settings = pymodm.fields.DictField()
    message = pymodm.fields.CharField()

    @property
    def identifier(self) -> bson.ObjectId:
        """
        Get the identifier for this trial result
        :return:
        """
        return self._id

    @abc.abstractmethod
    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this system. Pass these to "get_properties", below.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        :param columns:
        :return:
        """
        pass
