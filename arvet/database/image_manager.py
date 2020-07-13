import abc
import logging
import typing
import pathlib
import numpy as np
import h5py
import xxhash


class ImageGroup(metaclass=abc.ABCMeta):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    @abc.abstractmethod
    def can_write(self) -> bool:
        pass

    @abc.abstractmethod
    def get_image(self, path: str) -> np.ndarray:
        """
        Get a particular image based on a path.
        Paths are stored in the database, and are used to
        Groups are pre-determined when
        :param path: The path of the image to get
        :return: The image stored at the given path, as a numpy array
        :raises: KeyError if the given path does not match any image
        """
        pass

    @abc.abstractmethod
    def is_valid_path(self, path: str) -> bool:
        """
        Check if the given path matches an image within the group.
        Paths that fail this check will raise KeyError when retrieved using get
        :param path:
        :return:
        """
        pass

    @abc.abstractmethod
    def store_image(self, data: np.ndarray) -> str:
        """
        Store an image, within a particular gorup
        :param data: The image data to store
        :return: The path to retrieve that image from that group. Use that to retrieve the image later
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


class ImageManager(metaclass=abc.ABCMeta):
    """
    A class to manage storing images on disk, outisde the database.
    """

    @abc.abstractmethod
    def get_group(self, group_name: str, allow_write: bool = False) -> ImageGroup:
        """
        Get a particular group of images to read and write from
        This is a file handle
        :param group_name: The name of the group. Different group names give different groups with different image data
        :param allow_write: Are we going to try and write to the image file?
        :return:
        """
        pass


# ----- Actual Implementation -----


class HDF5Group(ImageGroup):

    def __init__(self, file_path: pathlib.Path, lock_file: pathlib.Path, allow_write: bool = False):
        self._file_path = pathlib.Path(file_path)
        self._lock_file = pathlib.Path(lock_file)
        self._contexts = 0
        self._storage = None
        self._allow_write = bool(allow_write)
        self._has_lock = False

    def __enter__(self):
        if self._storage is None:
            if self._allow_write:
                # Open the file for writing.
                # No other process may be accessing the file, we are the single writer
                if self._lock_file.exists():
                    # Someone else is writing. Fail.
                    raise IOError(f"Cannot open {self._file_path}, another process is currently writing.")
                else:
                    self._has_lock = True
                    self._lock_file.touch()
                self._storage = h5py.File(self._file_path, 'a', libver='latest')
            else:
                # Open the file for reading only
                # Other processes may also open the file.
                self._storage = h5py.File(self._file_path, 'r', libver='latest', swmr=True)
        self._contexts += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._contexts -= 1
        if self._contexts <= 0:
            if self._storage is not None:
                self._storage.close()
            if self._has_lock:
                # We set a lock file to flag other processes. Clear it when we're done
                self._lock_file.unlink()
            self._storage = None
            self._contexts = 0

    def can_write(self) -> bool:
        return self._allow_write

    def get_image(self, path: str) -> typing.Union[np.ndarray, None]:
        if path is not None:
            with self:
                data = self._storage.get(path, default=None)
                if data is not None:
                    return np.array(data)
        return None

    def is_valid_path(self, path: str) -> bool:
        if path is not None:
            with self:
                return path in self._storage
        return False

    def store_image(self, data: np.ndarray) -> str:
        # Use a context to open our file if it is not already open
        with self:
            # Try different seeds until we find one that doesn't collide,
            # or we find that this image is already stored.
            # Since the iteration is in the same order every time, and we check equality,
            # will always resolve collisions if it is possible to do so.
            for seed in range(2 ** 32):
                path = self.find_path_for_image(data, seed)
                if path not in self._storage:
                    if self._allow_write:
                        # Hash value is available, store the data there
                        self._storage[path] = data
                        return path
                    # We reached the point we'd write to the file, but we are not configured to do so. Fail.
                    raise RuntimeError("ImageManager not configured for writing")
                else:
                    try:
                        existing_data = self._storage[path]
                    except KeyError as exp:
                        logging.getLogger(__name__).error(
                            f"Error reading data for key {path}, derived with seed '{seed}'")
                        raise exp
                    if np.array_equal(data, existing_data):
                        # This image is already stored, return the path
                        return path
        raise RuntimeError("Could not find a seed that allows this image to be stored in the database")

    def remove_image(self, path: str):
        if self._allow_write:
            with self:
                if path in self._storage:
                    del self._storage[path]
                    return True
            return False
        raise RuntimeError("ImageManager not configured for writing")

    @staticmethod
    def find_path_for_image(data, seed=0):
        """
        Find a valid path to store this image, based on a hash of the
        :param data:
        :param seed: A seed passed to the hash, for collision handling.
        :return:
        """
        digest = xxhash.xxh64(data, seed=seed).hexdigest()
        return digest


class DefaultImageManager(ImageManager):
    """
    The default implementation of the ImageManager class, uses hdf5 to store image files, referenced by hash.
    This is what will be created by the configure method below.
    """

    def __init__(self, root: typing.Union[str, pathlib.Path]):
        self._root = pathlib.Path(root)
        self.open_groups = dict()

    def get_group(self, group_name: str, allow_write: bool = False) -> ImageGroup:
        """

        :param group_name:
        :param allow_write:
        :return:
        """
        if group_name not in self.open_groups:
            self.open_groups[group_name] = HDF5Group(
                file_path=self._root / (group_name + '.hdf5'),
                lock_file=self._root / (group_name + '.lock'),
                allow_write=allow_write
            )
        group = self.open_groups[group_name]
        if allow_write and not group.can_write:
            raise IOError(f"Cannot write to group {group_name}, it is already open read-only")
        return group


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
    image_manager = DefaultImageManager(root=config['path'])
    set_image_manager(image_manager)


def set_image_manager(image_manager: ImageManager) -> None:
    global __image_manager
    __image_manager = image_manager


def get() -> ImageManager:
    """
    Get the Image Manager
    :return:
    """
    global __image_manager
    if __image_manager is None:
        raise RuntimeError("No image manager configured, did you forget to call 'configure'?")
    return __image_manager
