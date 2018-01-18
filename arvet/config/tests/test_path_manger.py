import unittest
import os
import arvet.config.path_manager as pm


class TestPathManger(unittest.TestCase):

    def test_gets_absolute_path(self):
        subject = pm.PathManager(['.'])
        result = subject.find_path('.')
        self.assertEqual(os.path.abspath('.'), result)

    def test_expands_home_dir(self):
        subject = pm.PathManager(['~'])
        result = subject.find_path('.')
        self.assertEqual(os.path.abspath(os.path.expanduser('~')), result)

    def test_check_dir(self):
        subject = pm.PathManager([find_project_root()])
        self.assertTrue(subject.check_dir('arvet/config'))
        self.assertFalse(subject.check_dir('arvet/config/path_manager.py'))
        self.assertFalse(subject.check_dir('notafolder'))

    def test_find_dir(self):
        root_dir = find_project_root()
        subject = pm.PathManager([root_dir])
        self.assertEqual(os.path.join(root_dir, 'arvet', 'config'), subject.find_dir('arvet/config'))
        with self.assertRaises(FileNotFoundError):
            subject.find_dir('arvet/config/path_manager.py')
        with self.assertRaises(FileNotFoundError):
            subject.find_dir('notafolder')

    def test_check_file(self):
        subject = pm.PathManager([find_project_root()])
        self.assertFalse(subject.check_file('arvet/config'))
        self.assertTrue(subject.check_file('arvet/config/path_manager.py'))
        self.assertFalse(subject.check_file('notafile'))

    def test_find_file(self):
        root_dir = find_project_root()
        subject = pm.PathManager([root_dir])
        with self.assertRaises(FileNotFoundError):
            subject.find_file('arvet/config')
        self.assertEqual(os.path.join(root_dir, 'arvet', 'config', 'path_manager.py'),
                         subject.find_file('arvet/config/path_manager.py'))
        with self.assertRaises(FileNotFoundError):
            subject.find_file('notafile')

    def test_check_path(self):
        subject = pm.PathManager([find_project_root()])
        self.assertTrue(subject.check_path('arvet/config'))
        self.assertTrue(subject.check_path('arvet/config/path_manager.py'))
        self.assertFalse(subject.check_path('notafolder'))

    def test_find_path(self):
        root_dir = find_project_root()
        subject = pm.PathManager([root_dir])
        self.assertEqual(os.path.join(root_dir, 'arvet', 'config'), subject.find_path('arvet/config'))
        self.assertEqual(os.path.join(root_dir, 'arvet', 'config', 'path_manager.py'),
                         subject.find_file('arvet/config/path_manager.py'))
        with self.assertRaises(FileNotFoundError):
            subject.find_path('notafolder')


def find_project_root():
    """
    We're using the project structure for tests,
    find the directory that is the project root, traversing up from the current working dir.
    :return:
    """
    path = os.getcwd()
    while path is not '/':
        if os.path.isdir(os.path.join(path, 'arvet')) and \
                os.path.isdir(os.path.join(path, 'arvet', 'config')):
            return path
        path = os.path.abspath(os.path.join(path, '..'))    # Move up a folder level
    raise FileNotFoundError("Could not find the project root, run this test from somewhere within the project")
