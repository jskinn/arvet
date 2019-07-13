import unittest
import unittest.mock as mock
from os.path import abspath
from arvet.batch_analysis.job_systems.hpc_job_system import HPCJobSystem
from arvet.batch_analysis.job_systems.simple_job_system import SimpleJobSystem
import arvet.batch_analysis.job_systems.job_system_factory as job_system_factory


class TestJobSystemFactory(unittest.TestCase):

    @mock.patch('arvet.batch_analysis.job_systems.job_system_factory.HPCJobSystem', autospec=HPCJobSystem)
    def test_create_hpc_job_system(self, mock_hpc_constructor):
        mock_hpc_constructor.side_effect = lambda config, config_file: HPCJobSystem(config, config_file)
        job_system_config = {
            'job_system': 'hpc',
            'node_id': 'my-test-node',
            'task_config': {
                'allow_import_dataset': False,
                'allow_run_system': False,
                'allow_measure': True,
                'allow_trial_comparison': True
            },
            'environment': '/my/virtualenv',
            'job_location': '/my/job/storage',
            'job_name_prefix': '/'
        }
        job_system = job_system_factory.create_job_system({'job_system_config': job_system_config}, 'myconfig.yml')

        self.assertIsInstance(job_system, HPCJobSystem)
        self.assertTrue(mock_hpc_constructor.called)
        self.assertEqual(mock.call(job_system_config, config_file='myconfig.yml'), mock_hpc_constructor.call_args)

    @mock.patch('arvet.batch_analysis.job_systems.job_system_factory.SimpleJobSystem', autospec=SimpleJobSystem)
    def test_create_simple_job_system(self, mock_simple_constructor):
        mock_simple_constructor.side_effect = lambda config, config_file: SimpleJobSystem(config, config_file)
        job_system_config = {
            'job_system': 'simple',
            'node_id': 'my-test-node',
            'task_config': {
                'allow_import_dataset': False,
                'allow_run_system': False,
                'allow_measure': True,
                'allow_trial_comparison': True
            }
        }
        job_system = job_system_factory.create_job_system({'job_system_config': job_system_config}, 'myconfig.yml')

        self.assertIsInstance(job_system, SimpleJobSystem)
        self.assertTrue(mock_simple_constructor.called)
        self.assertEqual(mock.call(job_system_config, config_file='myconfig.yml'), mock_simple_constructor.call_args)

    def test_is_case_insensitive(self):
        for job_system_type in [
            'hpC',
            'hPc',
            'hPC',
            'Hpc',
            'HpC',
            'HPc',
            'HPC'
        ]:
            job_system = job_system_factory.create_job_system({
                'job_system_config': {'job_system': job_system_type}
            }, 'myconfig.yml')
            self.assertIsInstance(job_system, HPCJobSystem)

    def test_creates_simple_job_system_by_default(self):
        job_system = job_system_factory.create_job_system({}, 'myconfig.yml')
        self.assertIsInstance(job_system, SimpleJobSystem)
        self.assertEqual(abspath('myconfig.yml'), job_system._config_path)
