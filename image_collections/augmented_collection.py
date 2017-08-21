import abc
import database.entity
import core.sequence_type
import core.image_source


class AugmentedImageCollection(core.image_source.ImageSource, database.entity.Entity):
    """
    A wrapper around an image collection, which performs data augmentation.
    This
    """

    def __init__(self, inner, augmenters=None, id_=None, **kwargs):
        """
        Create a new augmented collection, which wraps an image source and provides augmented images
        from that source. There may be multiple augmentations per image
        :param inner: The inner image source to wrap, it will be
        :param augmenters: A list of things to perform data augmentation, or None for unchanged images
        """
        super().__init__(id_=id_, **kwargs)
        self._inner = inner
        self._augmenters = tuple(aug for aug in augmenters if aug is None or
                                 (hasattr(aug, 'augment') and callable(aug.augment) and
                                  hasattr(aug, 'serialize') and callable(aug.serialize)))
        self._current_image = None
        self._current_timestamp = None
        self._aug_index = 0

    def __len__(self):
        return len(self._inner) * len(self._augmenters)

    def __getitem__(self, item):
        return self.get(item)

    @property
    def is_stored_in_database(self):
        return self._inner.is_stored_in_database

    @property
    def is_stereo_available(self):
        return self._inner.is_stereo_available

    @property
    def is_normals_available(self):
        return self._inner.is_normals_available

    @property
    def sequence_type(self):
        if len(self._augmenters) > 1:
            # Each image appears multiple times, it becomes non-sequential
            return core.sequence_type.ImageSequenceType.NON_SEQUENTIAL
        return self._inner.sequence_type

    @property
    def is_per_pixel_labels_available(self):
        return self._inner.is_per_pixel_labels_available

    @property
    def is_depth_available(self):
        return self._inner.is_depth_available

    @property
    def is_labels_available(self):
        return self._inner.is_depth_available

    @property
    def supports_random_access(self):
        return self._inner.supports_random_access

    def get_camera_intrinsics(self):
        return self._inner.get_camera_intrinsics()

    def get(self, index):
        """
        Get the image at a particular index
        Loops
        :param index:
        :return:
        """
        if not self.supports_random_access:
            raise ValueError("This collection does not support random access")
        elif 0 <= index < len(self):
            inner_idx = index // len(self._augmenters)
            aug_idx = index - inner_idx * len(self._augmenters)
            return apply_augmenter(self._inner[inner_idx], self._augmenters[aug_idx])
        else:
            raise IndexError("Index {0} is out of range".format(index))

    def begin(self):
        """
        Start all the image sources, we're ready to start returning images.
        :return: void
        """
        self._inner.begin()
        self._current_image, self._current_timestamp = self._inner.get_next_image()
        if len(self._augmenters) > 1:
            self._current_timestamp = 0
        self._aug_index = 0

    def get_next_image(self):
        if self.is_complete():
            return None
        im = apply_augmenter(self._current_image, self._augmenters[self._aug_index])
        # Looping with a single augment means this is self._current_timestamp * 1 + 0 always
        timestamp = self._current_timestamp * len(self._augmenters) + self._aug_index
        self._aug_index += 1
        if self._aug_index >= len(self._augmenters) and not self._inner.is_complete():
            self._aug_index = 0
            if len(self._augmenters) > 1:
                self._current_image, _ = self._inner.get_next_image()
                self._current_timestamp += 1
            else:
                self._current_image, self._current_timestamp = self._inner.get_next_image()
        return im, timestamp

    def is_complete(self):
        return self._aug_index >= len(self._augmenters) and self._inner.is_complete()

    def serialize(self):
        serialized = super().serialize()
        serialized['inner'] = self._inner.identifier
        serialized['augmenters'] = [aug.serialize() if aug is not None else None for aug in self._augmenters]
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'inner' in serialized_representation:
            s_inner = db_client.image_source_collection.find_one({'_id': serialized_representation['inner']})
            kwargs['inner'] = db_client.deserialize_entity(s_inner)
        if 'augmenters' in serialized_representation:
            kwargs['augmenters'] = [db_client.deserialize_entity(s_aug) if s_aug is not None else None
                                    for s_aug in serialized_representation['augmenters']]
        return super().deserialize(serialized_representation, db_client, **kwargs)


def apply_augmenter(image, augmenter):
    if callable(augmenter):
        return augmenter(image)
    elif hasattr(augmenter, 'augment') and callable(augmenter.augment):
        return augmenter.augment(image)
    return image


class ImageAugmenter(database.entity.Entity, metaclass=database.entity.AbstractEntityMetaclass):
    """
    A simple base class for things that do image augmentation.
    Some have properties, and we need to be able to ave and load them from the database,
    which is why they're not just function pointers.
    """

    @abc.abstractmethod
    def augment(self, image):
        """
        Produce a new image object based on an existing image.
        :param image:
        :return:
        """
        pass
