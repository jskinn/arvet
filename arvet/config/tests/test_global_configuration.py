# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os
import time
import yaml
import arvet.config.global_configuration as global_conf


class TestGlobalConfiguration(unittest.TestCase):

    def test_save_global_config_writes_to_file(self):
        mock_open = mock.mock_open()
        filename = 'test_config_file_1'
        with mock.patch('arvet.config.global_configuration.open', mock_open, create=True):
            global_conf.save_global_config(filename, {'a': 1})
        self.assertTrue(mock_open.called)
        self.assertEqual(filename, mock_open.call_args[0][0])

    @mock.patch('arvet.config.global_configuration.yaml.load', autospec=yaml.load)
    @mock.patch('arvet.config.global_configuration.os.path.isfile', autospec=os.path.isfile)
    def test_load_global_config_reads_config_file_if_available(self, mock_isfile, mock_yaml_load):
        mock_isfile.return_value = True
        mock_yaml_load.return_value = {}
        mock_open = mock.mock_open()
        filename = 'test_config_file_2'
        with mock.patch('arvet.config.global_configuration.open', mock_open, create=True):
            global_conf.load_global_config(filename)
        self.assertTrue(mock_open.called)
        self.assertEqual(filename, mock_open.call_args[0][0])

    @mock.patch('arvet.config.global_configuration.save_global_config', autospec=global_conf.save_global_config)
    @mock.patch('arvet.config.global_configuration.yaml.load', autospec=yaml.load)
    @mock.patch('arvet.config.global_configuration.os.path.isfile', autospec=os.path.isfile)
    def test_load_global_config_saves_config_file_if_not_available(self, mock_isfile, mock_yaml_load, mock_save):
        mock_isfile.return_value = False
        mock_yaml_load.return_value = {}
        filename = 'test_config_file_3'
        global_conf.load_global_config(filename)
        self.assertTrue(mock_save.called)
        self.assertEqual(filename, mock_save.call_args[0][0])

    @mock.patch('arvet.config.global_configuration.save_global_config', autospec=global_conf.save_global_config)
    @mock.patch('arvet.config.global_configuration.yaml.load', autospec=yaml.load)
    @mock.patch('arvet.config.global_configuration.os.path.isfile', autospec=os.path.isfile)
    def test_load_global_config_does_not_save_config_file_if_available(self, mock_isfile, mock_yaml_load, mock_save):
        mock_isfile.return_value = True
        mock_yaml_load.return_value = {}
        filename = 'test_config_file_4'
        with mock.patch('arvet.config.global_configuration.open', mock.mock_open(), create=True):
            global_conf.load_global_config(filename)
        self.assertFalse(mock_save.called)

    @mock.patch('arvet.config.global_configuration.time.sleep', autospec=time.sleep)
    @mock.patch('arvet.config.global_configuration.yaml.load', autospec=yaml.load)
    @mock.patch('arvet.config.global_configuration.os.path.isfile', autospec=os.path.isfile)
    def test_load_global_config_waits_and_retries_three_times_if_load_failed(self, mock_isfile, mock_yaml_load,
                                                                             mock_sleep):
        mock_isfile.return_value = True
        mock_yaml_load.return_value = None
        with mock.patch('arvet.config.global_configuration.open', mock.mock_open(), create=True):
            global_conf.load_global_config('test_config_file_5')
        self.assertEqual(3, mock_yaml_load.call_count)
        self.assertEqual(3, mock_sleep.call_count)

    @mock.patch('arvet.config.global_configuration.yaml.load', autospec=yaml.load)
    @mock.patch('arvet.config.global_configuration.os.path.isfile', autospec=os.path.isfile)
    def test_load_global_config_returns_read_config_merged_with_defaults(self, mock_isfile, mock_yaml_load):
        mock_isfile.return_value = True
        config = {
            'test': 12.35,
            'database_config': {
                'database_name': 'a_different_database',
                'gridfs_bucket': 'file_system_fs',
                'collections': {
                    'trainer_collection': 'these_are_the_trainers_yo',
                    'system_collection': 'deze_sysTems',
                    'benchmarks_collection': 'mark_those_benches',
                }
            },
            'job_system_config': {
                'a': 1
            },
            'logging': {
                'demo': 'ATestProperty'
            }
        }
        mock_yaml_load.return_value = config
        with mock.patch('arvet.config.global_configuration.open', mock.mock_open(), create=True):
            result = global_conf.load_global_config('test_config_file_6')
        for key, val in config.items():
            self.assertIn(key, result)
            if isinstance(val, dict):
                for inner1_key, inner1_val in val.items():
                    self.assertIn(inner1_key, result[key])
                    if isinstance(inner1_val, dict):
                        for inner2_key, inner2_val in inner1_val.items():
                            self.assertIn(inner2_key, result[key][inner1_key])
                            self.assertEqual(inner2_val, result[key][inner1_key][inner2_key])
                    else:
                        self.assertEqual(inner1_val, result[key][inner1_key])
            else:
                self.assertEqual(val, result[key])
