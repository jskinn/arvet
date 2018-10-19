import abc
import pymodm


class ABCModelMeta(pymodm.base.models.TopLevelMongoModelMetaclass, abc.ABCMeta):
    """
    A quick helper superclass to allow pymodm Models which are also abstract base classes.
    """
    pass
