import pymodm
from pymodm import validators


class EnumField(pymodm.fields.MongoBaseField):
    """
    A custom pymodm field for enums, restricts values to values of a particular enum type.
    Will be stored in the database as the string enum name, for better human readability.
    If the storage becomes a problem, switch to storing the integer value, but it shouldn't be an issue.
    """

    def __init__(self, enum_type, verbose_name=None, mongo_name=None, **kwargs):
        """

        :param enum_type: The enum type to store. Valid values must be an instance of this type.
        :param verbose_name: The human-readable name of this field
        :param mongo_name: The name of this field in mongodb
        :param kwargs: Additional kwargs passed to MongoBaseField
        """
        super(EnumField, self).__init__(
            verbose_name=verbose_name,
            mongo_name=mongo_name,
            choices=[e for e in enum_type],
            **kwargs
        )
        self.enum_type = enum_type
        self.validators.append(validators.validator_for_type(enum_type))

    def to_python(self, value):
        if isinstance(value, self.enum_type):
            return value
        return self.enum_type[value]

    def to_mongo(self, value):
        if isinstance(value, self.enum_type):
            return value.name
        return value
