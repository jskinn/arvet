import pymodm
from pymodm import validators
from pymodm.queryset import QuerySet
from pymodm.manager import Manager
import numpy as np
import arvet.database.image_manager


class ImageField(pymodm.fields.MongoBaseField):
    """
    A field containing an image, which is a numpy array.
    Images are stored outside the database, managed by the ImageMangager
    (see arvet.database.image_manager)
    """

    def __init__(self, group='', verbose_name=None, mongo_name=None, **kwargs):
        """

        :param group: A subgroup to store the image under
        :param verbose_name: The human-readable name of this field
        :param mongo_name: The name of this field in mongodb
        :param kwargs: Additional kwargs passed to MongoBaseField
        """
        super(ImageField, self).__init__(
            verbose_name=verbose_name,
            mongo_name=mongo_name,
            **kwargs
        )
        self._group = group
        self.validators.append(validators.validator_for_type(np.ndarray))

    def __set__(self, inst, value):
        if isinstance(value, np.ndarray):
            # Numpy arrays are python values, set as such. Otherwise, they will set as mongo values
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
                path = image_manager.store_image(value, group=self._group)
            return path
        return value


class ImageQuerySet(QuerySet):
    """
    A custom query set for models that use image fields.
    Must use this to suport deleting images from the image manager
    when the corresponding model is deleted.
    """

    def delete(self):
        """
        Delete all the models in the query set.
        Models delegate to this
        :return:
        """
        # Search the model, and all embedded models for ImageFields
        image_paths = []
        to_search = [
            (self._model, [doc for doc in self.values()])
        ]
        while len(to_search) > 0:
            model, docs = to_search.pop()

            # Find all the Image fields on this model
            fields = model._mongometa.get_fields()
            image_fields = [
                field.attname
                for field in fields
                if isinstance(field, ImageField)
            ]

            # Get the values of those fields from the documents
            image_paths.extend(doc[name] for name in image_fields for doc in docs)

            # Look in embedded documents as well
            to_search.extend(
                (field.related_model, [doc[field.attname] for doc in docs])
                for field in fields
                if isinstance(field, pymodm.fields.EmbeddedDocumentField)
            )
            to_search.extend(
                (field.related_model, [inner_doc for doc in docs for inner_doc in doc[field.attname]])
                for field in fields
                if isinstance(field, pymodm.fields.EmbeddedDocumentListField)
            )

        # Delete all the images at those paths
        if len(image_paths) > 0:
            with arvet.database.image_manager.get() as image_manager:
                for path in image_paths:
                    image_manager.remove_image(path)

        # Proceed with the rest of the delete
        super(ImageQuerySet, self).delete()


# Custom manager using the ImageQueryset by default.
# Assign this as the manager for models with image fields
ImageManager = Manager.from_queryset(ImageQuerySet)

