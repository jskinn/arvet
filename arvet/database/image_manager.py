import abc
import typing
import numpy as np
import h5py
import xxhash


class ImageManager(metaclass=abc.ABCMeta):
    """
    A class to manage storing images on disk, outisde the database.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def get_image(self, path: str) -> typing.Union[np.ndarray, None]:
        """
        Get a particular image based on a path.
        Paths are stored in the database,
        :param path: A path
        :return:
        """
        pass

    @abc.abstractmethod
    def is_valid_path(self, path: str) -> bool:
        """

        :param path:
        :return:
        """
        pass

    @abc.abstractmethod
    def store_image(self, data: np.ndarray, group: str = '') -> str:
        """
        Store an image.
        :param data: The image data to store
        :param group:
        :return:
        """
        pass

    @abc.abstractmethod
    def remove_image(self, path: str) -> bool:
        """
        Remove an image from the store
        :param path:
        :return:
        """
        pass


class DefaultImageManager(ImageManager):
    """
    The default implementation of the ImageManager class, uses hdf5 to store image files, referenced by hash.
    This is what will be created by the configure method below.
    """

    def __init__(self, file_path, group_name=''):
        self._file_path = file_path
        self._prefix = group_name.strip('/')
        self._contexts = 0
        self._storage = None

    def __enter__(self):
        if self._storage is None:
            self._storage = h5py.File(self._file_path, 'a')
        self._contexts += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._contexts -= 1
        if self._contexts <= 0:
            if self._storage is not None:
                self._storage.close()
            self._storage = None
            self._contexts = 0

    def get_image(self, path: str) -> typing.Union[np.ndarray, None]:
        with self:
            data = self._storage.get(path, default=None)
            if data is not None:
                return np.array(data)
        return None

    def is_valid_path(self, path: str) -> bool:
        with self:
            return path in self._storage

    def store_image(self, data: np.ndarray, group: str = '') -> str:
        # Use a context to open our file if it is not already open
        with self:
            # Try different seeds until we find one that doesn't collide, or we find that this image is already stored
            # Since the iteration is in the same order every time, and we check equality, will always resolve collisions
            # if it is possible to do so.
            for seed in range(2 ** 32):
                path = self.find_path_for_image(data, group, seed)
                if path not in self._storage:
                    # Hash value is available, store the data there
                    self._storage[path] = data
                    return path
                elif np.array_equal(data, self._storage[path]):
                    # This image is already stored, return the path
                    return path
        raise RuntimeError("Could not find a seed that allows this image to be stored in the database")

    def remove_image(self, path: str):
        with self:
            if path in self._storage:
                del self._storage[path]
                return True
        return False

    def find_path_for_image(self, data, group='', seed=0):
        """
        Find a valid path to store this image, based on a hash of the
        :param data:
        :param group:
        :param seed: A seed passed to the hash, for collision handling.
        :return:
        """
        group = group.strip('/')
        digest = xxhash.xxh64(data, seed=seed).hexdigest()
        return '/'.join(part for part in (self._prefix, group, digest) if len(part) > 0)


# ----- Global Singleton -----
# There is a single global image manager instance,
# which is used when loading images from the database
# It must be initialized, either from config, or explicitly with an image manager object
# This is the best solution I have to inject configuration and custom loading at database deserialization time
__image_manager = None


def configure(config) -> None:
    """
    Load the default image manager with some configuration
    :param config:
    :return:
    """
    if 'image_manager' in config:
        config = config['image_manager']
    image_manager = DefaultImageManager(
        file_path=config['path'],
        group_name=config.get('group', '')
    )
    set_image_manager(image_manager)


def set_image_manager(image_manager: ImageManager) -> None:
    global __image_manager
    __image_manager = image_manager


def get() -> ImageManager:
    """
    Get the Image Manager
    :return:
    """
    if __image_manager is None:
        raise RuntimeError("No image manager configured, did you forget to call 'configure'?")
    return __image_manager
