import pymodm
import pymodm.validators
import arvet.util.transform as tf


class TransformField(pymodm.fields.MongoBaseField):
    """
    A custom pymodm field for transforms.
    A transform is a 3D pose, including both translation and orientation
    see also arvet.uti.transform
    """

    def __init__(self, verbose_name=None, mongo_name=None, **kwargs):
        """

        :param verbose_name: The human-readable name of this field
        :param mongo_name: The name of this field in mongodb
        :param kwargs: Additional kwargs passed to MongoBaseField
        """
        super(TransformField, self).__init__(
            verbose_name=verbose_name,
            mongo_name=mongo_name,
            **kwargs
        )
        self.validators.append(pymodm.validators.validator_for_type(tf.Transform))

    def to_python(self, value):
        if isinstance(value, tf.Transform):
            return value
        return tf.Transform.deserialize(value)

    def to_mongo(self, value):
        if isinstance(value, tf.Transform):
            return value.serialize()
        return value
