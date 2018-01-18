import typing
import os


# TODO: Once we standardize on 3.6, use pathlib
class PathManager:
    """
    A simple class to allow node-independent location of files and folders.
    A list of folders to search is specified in the config file,
    and then all paths are expressed relative to those folders.
    At run time, we search all of the configured base directories for matching files or folders,
    and return the first potential match.
    """

    def __init__(self, paths: typing.List[str]):
        paths = [os.path.abspath(os.path.expanduser(path)) for path in paths]
        self._roots = [path for path in paths if os.path.isdir(path)]

    def check_dir(self, directory: str) -> bool:
        """
        Check if a given directory can be found by the path manager
        :param directory: The directory to check
        :return: True iff the directory can be found, false otherwise
        """
        return find_absolute_path(directory, self._roots, os.path.isdir) is not None

    def find_dir(self, directory: str) -> str:
        """
        Find a particular directory. Raises FileNotFoundError if it doesn't exist.
        :param directory: The directory to find, relative to one of the configured base folders
        :return: The full path to the given directory
        """
        full_path = find_absolute_path(directory, self._roots, os.path.isdir)
        if full_path is None:
            raise FileNotFoundError("Could not find directory {0} within {1}".format(directory, str(self._roots)))
        return full_path

    def check_file(self, file: str) -> bool:
        """
        Check if a given file can be found by the path manager
        :param file: The file path, relative to one of the configured root folders
        :return: True iff the file exists and is a file (not a directory)
        """
        return find_absolute_path(file, self._roots, os.path.isfile) is not None

    def find_file(self, file: str) -> str:
        """
        Find a particular file. Raises FileNotFoundError if it doesn't exist.
        :param file: The file to find, relative to one of the configured
        :return: The absolute path
        """
        full_path = find_absolute_path(file, self._roots, os.path.isfile)
        if full_path is None:
            raise FileNotFoundError("Could not find file {0} within {1}".format(file, str(self._roots)))
        return full_path

    def check_path(self, relative_path: str) -> bool:
        """
        Check if a given path (file or directory) exists and can be found by the path manager
        :param relative_path: The relative path
        :return: True iff the relative path can be found
        """
        return find_absolute_path(relative_path, self._roots, os.path.exists) is not None

    def find_path(self, relative_path: str) -> str:
        """
        Find either a file or directory
        :param relative_path: The relative path of file or folder to find
        :return: The full path on this platform
        """
        full_path = find_absolute_path(relative_path, self._roots, os.path.exists)
        if full_path is None:
            raise FileNotFoundError("Could not find file or directory {0} within {1}".format(relative_path,
                                                                                             str(self._roots)))
        return full_path


def find_absolute_path(path: str, roots: typing.List[str], filter_func: typing.Callable[[str], bool]) \
        -> typing.Union[str, None]:
    """
    A helper to do the heavy lifting for the path manager.
    Finds full paths that match a certain function, usually one of isdir, isfile, or exists
    :param path: A relative path to search for
    :param roots: A list of base paths to check
    :param filter_func: A filter to check the full path
    :return: The full path, if it exists, or None if it doesnt
    """
    for root in roots:
        full_path = os.path.join(root, path)
        if filter_func(full_path):
            return os.path.abspath(full_path)
    return None
