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

    def find_dir(self, directory: str) -> str:
        """
        Find a particular directory. Raises FileNotFoundError if it doesn't exist.
        :param directory: The directory to find, relative to one of the configured base folders
        :return:
        """
        for path in self._roots:
            full_path = os.path.join(path, directory)
            if os.path.isdir(full_path):
                return full_path
        raise FileNotFoundError("Could not find directory {0} within {1}".format(directory, str(self._roots)))

    def find_file(self, file: str) -> str:
        """
        Find a particular file. Raises FileNotFoundError if it doesn't exist.
        :param file: The file to find, relative to one of the configured
        :return:
        """
        for path in self._roots:
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path):
                return full_path
        raise FileNotFoundError("Could not find file {0} within {1}".format(file, str(self._roots)))

    def find_path(self, relative_path: str) -> str:
        """
        Find either a file or directory
        :param relative_path:
        :return:
        """
        for path in self._roots:
            full_path = os.path.join(path, relative_path)
            if os.path.exists(full_path):
                return full_path
        raise FileNotFoundError("Could not find file or directory {0} within {1}".format(relative_path,
                                                                                         str(self._roots)))
