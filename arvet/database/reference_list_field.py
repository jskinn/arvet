from pymodm.fields import ListField, ReferenceField


class ReferenceListField(ListField):
    """
    A field that stores a list of references.
    Patched to register delete rules properly
    """

    def __init__(self, model, on_delete=ReferenceField.DO_NOTHING,
                 verbose_name=None, mongo_name=None,
                 **kwargs):
        """
        :parameters:
          - `verbose_name`: A human-readable name for the Field.
          - `mongo_name`: The name of this field when stored in MongoDB.
          - `field`: The Field instance of all items in this list.
            This needs to be an *instance* of a `MongoBaseField` subclass.

        .. seealso:: constructor for
                     :class:`~pymodm.base.fields.MongoBaseField`
        """
        super(ReferenceListField, self).__init__(
            field=ReferenceField(model, on_delete=on_delete),
            verbose_name=verbose_name,
            mongo_name=mongo_name,
            **kwargs
        )
        self._on_delete = on_delete

    def contribute_to_class(self, cls, name):
        super(ReferenceListField, self).contribute_to_class(cls, name)
        # Let inner fields know what model we're attached to.
        field = self._field

        if ReferenceField.DO_NOTHING != self._on_delete:
            field.related_model.register_delete_rule(self.model, name, self._on_delete)
