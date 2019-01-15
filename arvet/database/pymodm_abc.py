import abc
import pymodm


class ABCModelMeta(abc.ABCMeta, pymodm.base.models.TopLevelMongoModelMetaclass):
    """
    A quick helper superclass to allow pymodm Models which are also abstract base classes.
    """
    pass
