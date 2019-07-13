import pymodm


class TestBaseModel(pymodm.MongoModel):
    """
    A pointless monogmodel, that exists in a different module.
    Used to test module autoloading, see test_autoload_modules
    """
    pass
