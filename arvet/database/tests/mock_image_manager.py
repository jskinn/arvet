import numpy as np
import xxhash
import arvet.database.image_manager as im_manager


class MockImageGroup(im_manager.ImageGroup):

    def __init__(self, allow_write=True):
        self._storage = dict()
        self._allow_write = bool(allow_write)

    @property
    def can_write(self) -> bool:
        return self._allow_write

    def get_image(self, path: str) -> np.ndarray:
        return self._storage[path]

    def is_valid_path(self, path: str) -> bool:
        return path in self._storage

    def store_image(self, data: np.ndarray) -> str:
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
                existing_data = self._storage[path]
                if np.array_equal(data, existing_data):
                    # This image is already stored, return the path
                    return path
        raise RuntimeError("Could not find a seed that allows this image to be stored in the database")

    def remove_image(self, path: str) -> bool:
        """
        Remove an image from the store
        :param path:
        :return:
        """
        if path in self._storage:
            del self._storage[path]
            return True
        return False

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


class MockImageManager(im_manager.ImageManager):
    """
    A fake image manager that does not persist images, but holds them in memory.
    Use for tests.
    """

    def __init__(self):
        self.open_groups = dict()

    def get_group(self, group_name: str, allow_write: bool = False) -> im_manager.ImageGroup:
        """

        :param group_name:
        :param allow_write:
        :return:
        """
        if group_name not in self.open_groups:
            self.open_groups[group_name] = MockImageGroup(allow_write=allow_write)
        group = self.open_groups[group_name]
        if allow_write and not group.can_write:
            raise IOError(f"Cannot write to group {group_name}, it is already open read-only")
        return group
