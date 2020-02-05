import unittest
import os
from pathlib import Path
from arvet.config.path_manager import PathManager


class TestPathManger(unittest.TestCase):

    def test_gets_absolute_path(self):
        subject = PathManager(['.'])
        result = subject.find_path('.')
        self.assertEqual(Path('.').resolve(), result)

        subject = PathManager([Path('.')])
        result = subject.find_path('.')
        self.assertEqual(Path('.').resolve(), result)

    def test_expands_home_dir(self):
        subject = PathManager(['~'])
        result = subject.find_path('.')
        self.assertEqual(Path('~').expanduser().resolve(), result)

        subject = PathManager([Path('~')])
        result = subject.find_path('.')
        self.assertEqual(Path('~').expanduser().resolve(), result)

    def test_get_temp_folder_expands_users(self):
        subject = PathManager(['~'], '~')
        self.assertEqual(Path('~').expanduser(), subject.get_temp_folder())

    def test_get_temp_folder_returns_absolute_path(self):
        subject = PathManager(['~'], '.')
        self.assertEqual(Path('.').resolve(), subject.get_temp_folder())

    def test_get_output_dir_expands_home_dir(self):
        subject = PathManager(['~'], '~', output_dir='~')
        self.assertEqual(Path('~').expanduser(), subject.get_output_dir())

    def test_get_output_dir_uses_configured_dir(self):
        output_dir = (Path('~') / 'my_dir' / 'output').expanduser().resolve()
        subject = PathManager(['~'], '~', output_dir=str(output_dir))
        self.assertEqual(output_dir, subject.get_output_dir())

    def test_get_output_dir_defaults_to_first_path(self):
        first_path = Path(__file__).parent.resolve()
        subject = PathManager([str(first_path), '~'], '.')
        self.assertEqual(first_path, subject.get_output_dir())

    def test_get_output_dir_falls_back_to_first_path_if_output_path_not_within_any_path(self):
        first_path = Path(__file__).parent.resolve()
        subject = PathManager([str(first_path)], '.', output_dir=str(first_path.parent / 'myfolder'))
        self.assertEqual(first_path, subject.get_output_dir())

    def test_check_dir(self):
        subject = PathManager([find_project_root()])
        self.assertTrue(subject.check_dir('arvet/config'))
        self.assertTrue(subject.check_dir(Path('arvet/config')))
        self.assertFalse(subject.check_dir('arvet/config/path_manager.py'))
        self.assertFalse(subject.check_dir(Path('arvet/config/path_manager.py')))
        self.assertFalse(subject.check_dir('notafolder'))
        self.assertFalse(subject.check_dir(Path('notafolder')))

    def test_find_dir(self):
        root_dir = find_project_root()
        subject = PathManager([root_dir])
        self.assertEqual(root_dir / 'arvet' / 'config', subject.find_dir('arvet/config'))
        self.assertEqual(root_dir / 'arvet' / 'config', subject.find_dir(Path('arvet/config')))
        with self.assertRaises(FileNotFoundError):
            subject.find_dir('arvet/config/path_manager.py')
        with self.assertRaises(FileNotFoundError):
            subject.find_dir(Path('arvet/config/path_manager.py'))
        with self.assertRaises(FileNotFoundError):
            subject.find_dir('notafolder')
        with self.assertRaises(FileNotFoundError):
            subject.find_dir(Path('notafolder'))

    def test_check_file(self):
        subject = PathManager([find_project_root()])
        self.assertFalse(subject.check_file('arvet/config'))
        self.assertTrue(subject.check_file('arvet/config/path_manager.py'))
        self.assertFalse(subject.check_file('notafile'))

    def test_find_file(self):
        root_dir = find_project_root()
        subject = PathManager([root_dir])
        with self.assertRaises(FileNotFoundError):
            subject.find_file('arvet/config')
        self.assertEqual(root_dir / 'arvet' / 'config' / 'path_manager.py',
                         subject.find_file('arvet/config/path_manager.py'))
        with self.assertRaises(FileNotFoundError):
            subject.find_file('notafile')

    def test_check_path(self):
        subject = PathManager([find_project_root()])
        self.assertTrue(subject.check_path('arvet/config'))
        self.assertTrue(subject.check_path('arvet/config/path_manager.py'))
        self.assertFalse(subject.check_path('notafolder'))

    def test_find_path(self):
        root_dir = find_project_root()
        subject = PathManager([root_dir])
        self.assertEqual(root_dir / 'arvet' / 'config', subject.find_path('arvet/config'))
        self.assertEqual(root_dir / 'arvet' / 'config' / 'path_manager.py',
                         subject.find_file('arvet/config/path_manager.py'))
        with self.assertRaises(FileNotFoundError):
            subject.find_path('notafolder')


def find_project_root():
    """
    We're using the project structure for tests,
    find the directory that is the project root, traversing up from the current working dir.
    :return:
    """
    path = Path(os.getcwd())
    while path is not '/':
        if (path / 'arvet').is_dir() and (path / 'arvet' / 'config').is_dir():
            return path
        path = path.parent    # Move up a folder level
    raise FileNotFoundError("Could not find the project root, run this test from somewhere within the project")
