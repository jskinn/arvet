class no_auto_dereference(object):
    """
    Context manager that turns off automatic dereferencing.
    There is a bad interaction in the default

    Example::

        >>> some_profile = UserProfile.objects.first()
        >>> with no_auto_dereference(UserProfile):
        ...     some_profile.user
        ObjectId('5786cf1d6e32ab419952fce4')
        >>> some_profile.user
        User(name='Sammy', points=123)

    """

    def __init__(self, model, all_siblings=False):
        """
        :parameters:
          - `model`:  A :class:`~pymodm.MongoModel` class.
          - `all_siblings`: A

        """
        self.models = [model]
        self.orig_auto_deref = self.model._mongometa.auto_dereference

    def __enter__(self):
        self.model._mongometa.auto_dereference = False

    def __exit__(self, typ, val, tb):
        self.model._mongometa.auto_dereference = self.orig_auto_deref