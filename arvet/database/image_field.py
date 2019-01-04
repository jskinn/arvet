import pymodm
from pymodm import validators
import numpy as np
import arvet.database.image_manager


class ImageField(pymodm.fields.MongoBaseField):
    """
    A field containing an image, which is a numpy array.
    Images are stored outside the database, managed by the ImageMangager
    (see arvet.database.image_manager)
    """

    def __init__(self, verbose_name=None, mongo_name=None, **kwargs):
        """

        :param verbose_name: The human-readable name of this field
        :param mongo_name: The name of this field in mongodb
        :param kwargs: Additional kwargs passed to MongoBaseField
        """
        super(ImageField, self).__init__(
            verbose_name=verbose_name,
            mongo_name=mongo_name,
            **kwargs
        )
        self.validators.append(validators.validator_for_type(np.ndarray))

    def __set__(self, inst, value):
        if isinstance(value, np.ndarray):
            # Numpy arrays are python values, set as such. Otherwise, they will set as
            inst._data.set_python_value(self.attname, value)
        else:
            super(ImageField, self).__set__(inst, value)

    def is_blank(self, value):
        """Determine if the value is blank."""
        if isinstance(value, np.ndarray):
            # Custom handling for numpy arrays, which are hard to compare
            return value.size <= 0
        else:
            return super(ImageField, self).is_blank(value)

    def to_python(self, value):
        # Return immediately for blank values, or values that are already an image
        if self.is_blank(value) or isinstance(value, np.ndarray):
            return value

        if isinstance(value, str):
            with arvet.database.image_manager.get() as image_manager:
                return image_manager.get_image(value)
        return value

    def to_mongo(self, value):
        if isinstance(value, np.ndarray):
            with arvet.database.image_manager.get() as image_manager:
                path = image_manager.store_image(value)
            return path
        return value
