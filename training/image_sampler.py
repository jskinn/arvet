import random


class ImageSampler:
    """
    A wrapper for multiple image sources that allows random access,
    which performs sampling, training-validation segregation, and data augmentation.
    TODO: Replace this, it should be an image source as well.
    """

    def __init__(self, image_sources, augmenters=None, default_validation_fraction=0.3, loop=False):
        """
        Create a new sampler, with a bunch of settings
        A note about data augmentation, all augmenters have to be set at the start,
        I don't see any reason to regenerate the indexes for all existing image sources when new augments are available.
        :param image_sources: A collection of image sources to aggregate and sample from
        :param augmenters: A list of things to perform data augmentation,
        may either be callables or have an 'augment' method
        :param default_validation_fraction: The default fraction of images to use as validation set, defaults to 0.3
        :param loop: If true, the get methods will automatically loop the index into the valid range,
        so get(-1) returns the last element, get(n+1) returns the first, etc.
        """
        self._validation_fraction = default_validation_fraction
        self._current_index = 0
        self._loop = bool(loop)
        self._augmenters = [None]
        if augmenters is not None:
            self._augmenters += augmenters
        self._indexes = []
        self._validation_indexes = []
        self._image_sources = []
        for image_source in image_sources:
            self.add_image_source(image_source)

    @property
    def num_training(self):
        """
        The number of images in the training set
        :return:
        """
        return len(self._indexes)

    @property
    def num_validation(self):
        """
        The number of images in the validation set
        :return:
        """
        return len(self._validation_indexes)

    def get(self, index):
        """
        Get the image at a particular index
        This chooses randomly from one of the wrapped image sources
        :param index:
        :return:
        """
        if self._loop:
            index = index % len(self._indexes)
            if index < 0:
                index += len(self._indexes)
        if 0 <= index < len(self._indexes):
            return self._get_by_settings(*self._indexes[index])
        else:
            raise IndexError("Index {0} is out of range".format(index))

    def get_validation(self, index):
        """
        Get the image at a particular index
        This chooses randomly from one of the wrapped image sources
        :param index:
        :return:
        """
        if self._loop:
            index = index % len(self._validation_indexes)
            if index < 0:
                index += len(self._validation_indexes)
        if 0 <= index < len(self._validation_indexes):
            return self._get_by_settings(*self._validation_indexes[index])
        else:
            raise IndexError("Index {0} is out of range".format(index))

    def add_image_source(self, image_source, validation_fraction=None):
        """
        Add a new image source
        :param image_source: The image source to add
        :param validation_fraction: Override the fraction of this image source that will be used for validation
        :return:
        """
        # Duck typing!
        if hasattr(image_source, 'begin') and hasattr(image_source, 'timestamps') and (
                hasattr(image_source, '__getitem__') or hasattr(image_source, 'get')):
            num_new_indexes = len(image_source)
            if validation_fraction is None:
                validation_fraction = self._validation_fraction

            # Randomly partition the new images into validation and training sets.
            new_indexes = list(image_source.timestamps)
            random.shuffle(new_indexes)
            split = int(validation_fraction * num_new_indexes)

            # Add the new parameters to the training and validation sets
            self._validation_indexes += [(len(self._image_sources), new_indexes[idx], aug_idx)
                                         for idx in range(split) for aug_idx in range(len(self._augmenters))]
            self._indexes += [(len(self._image_sources), new_indexes[idx], aug_idx)
                              for idx in range(split, num_new_indexes) for aug_idx in range(len(self._augmenters))]
            self._image_sources.append(image_source)

    def shuffle(self, random_source=None):
        """
        Re-shuffle the images in this source to a new order
        Defaults to using the python random module for shuffling,
        but an alternative source of randomness may be passed as an argument.
        Does not shuffle the validation images, their order should be irrelevant.
        :param random_source: A random stream (such as the random module) with a shuffle module for shuffling a list
        :return: void
        """
        if random_source is not None and hasattr(random_source, 'shuffle'):
            random_source.shuffle(self._indexes)
        else:
            random.shuffle(self._indexes)

    def begin(self, shuffle=True):
        """
        Start all the image sources, we're ready to start taking images.
        :param shuffle: Should we shuffle the indexes?
        :return: void
        """
        for image_source in self._image_sources:
            image_source.begin()
        if shuffle:
            self.shuffle()

    def shutdown(self):
        """
        Shut down the image sources
        :return:
        """
        for image_source in self._image_sources:
            image_source.shutdown()

    def _get_by_settings(self, source_idx, idx, augmenter_idx):
        """
        Internal get, based on the different settings stored in the indexes lists.
        All the combinations of these settings are created, and are shuffled around
        in the indexes and validation indexes lists,
        :param source_idx: The index of the image
        :param idx:
        :param augmenter_idx:
        :return:
        """
        augmenter = self._augmenters[augmenter_idx]
        image = None
        if hasattr(self._image_sources[source_idx], 'get'):
            image = self._image_sources[source_idx].get(idx)
        elif hasattr(self._image_sources[source_idx], '__getitem__'):
            image = self._image_sources[source_idx][idx]
        if image is not None and augmenter is not None:
            if hasattr(augmenter, 'augment'):
                image = augmenter.augment(image)
            elif callable(augmenter):
                image = augmenter(image)
        return image
