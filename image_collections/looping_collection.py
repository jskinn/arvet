import database.entity
import core.sequence_type
import core.image_source


class LoopingCollection(core.image_source.ImageSource, database.entity.Entity):
    """
    A wrapper around an image collection, which makes it repeat.
    Very simple.
    """

    def __init__(self, inner, repeats=1, type_override=None, id_=None, **kwargs):
        """
        Create a looping collection wrapping an image source, and looping a specified number of times
        :param inner: The inner image source to wrap, it will be
        :param repeats: The total number of times to go through the image source. Default 1.
        :param type_override: Override the sequence type. Technically, looping is only sequential if the inner image
        source is sequential and it returns to where it started. We can't know that, so allow it to be set explicitly.
        """
        super().__init__(id_=id_, **kwargs)
        self._inner = inner
        self._repeats = max(int(repeats), 0)
        self._type_override = core.sequence_type.ImageSequenceType(type_override) if type_override is not None else None
        self._current_loop_count = 0
        self._max_timestamp = 0

    def __len__(self):
        return self._repeats * len(self._inner)

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
        if self._type_override is not None:
            return self._type_override
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

    def get_stereo_baseline(self):
        return self._inner.get_stereo_baseline()

    def begin(self):
        """
        Start all the image sources, we're ready to start returning images.
        :return: void
        """
        self._inner.begin()
        self._current_loop_count = 0

    def get_next_image(self):
        img, timestamp = self._inner.get_next_image()
        if timestamp > self._max_timestamp:
            self._max_timestamp = timestamp
        timestamp += self._current_loop_count * self._max_timestamp
        if self._inner.is_complete():
            self._current_loop_count += 1
            if self._current_loop_count < self._repeats:
                self._inner.begin()
        return img, timestamp

    def get(self, index):
        """
        Get the image at a particular index.
        :param index:
        :return:
        """
        if not self.supports_random_access:
            raise ValueError("This collection does not support random access")
        elif 0 <= index < len(self):
            index = index % len(self._inner)
            return self._inner[index]
        else:
            raise IndexError("Index {0} is out of range".format(index))

    def is_complete(self):
        return self._inner.is_complete() and self._current_loop_count >= self._repeats

    def serialize(self):
        serialized = super().serialize()
        serialized['inner'] = self._inner.identifier
        serialized['repeats'] = self._repeats
        serialized['type_override'] = self._type_override
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'inner' in serialized_representation:
            s_inner = db_client.image_source_collection.find_one({'_id': serialized_representation['inner']})
            kwargs['inner'] = db_client.deserialize_entity(s_inner)
        if 'repeats' in serialized_representation:
            kwargs['repeats'] = serialized_representation['repeats']
        if 'type_override' in serialized_representation:
            kwargs['type_override'] = serialized_representation['type_override']
        return super().deserialize(serialized_representation, db_client, **kwargs)
