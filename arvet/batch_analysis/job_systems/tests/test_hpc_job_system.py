# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os
import bson
from arvet.batch_analysis.task import Task
import arvet.batch_analysis.job_systems.hpc_job_system as hpc
import arvet.batch_analysis.scripts.run_task


class TestHPCJobSystem(unittest.TestCase):

    def test_works_with_empty_config(self):
        hpc.HPCJobSystem({})

    def test_cannot_generate_dataset(self):
        subject = hpc.HPCJobSystem({})
        self.assertFalse(subject.can_generate_dataset(bson.ObjectId(), {}))

    def test_configure_node_id(self):
        node_id = 'test-node-id-157890'
        subject = hpc.HPCJobSystem({'node_id': node_id})
        self.assertEqual(node_id, subject.node_id)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_script_checks_number_of_existing_jobs_if_a_limit_is_set(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({'max_jobs': 100})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.run_script('test_script', [])
        self.assertIn(mock.call(['qjobs'], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_script_does_not_check_number_of_existing_jobs_if_no_limit_is_set(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.run_script('test_script', [])
        self.assertNotIn(mock.call(['qjobs'], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    def test_run_script_wont_submit_more_scripts_than_job_limit(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'max_jobs': 10})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run, num_jobs=0)
                for _ in range(10):
                    mock_run.called = False
                    job_id = subject.run_script('test_script', [])
                    self.assertIsNotNone(job_id)
                    self.assertTrue(mock_run.called)
                mock_run.called = False
                self.assertIsNone(subject.run_script('test_script', []))
                self.assertFalse(mock_run.called)

    def test_existing_jobs_reduce_max_jobs(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({
            'max_jobs': 10
        })
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run, num_jobs=5)
                for _ in range(5):
                    mock_run.called = False
                    job_id = subject.run_script('test_script', [])
                    self.assertIsNotNone(job_id)
                    self.assertTrue(mock_run.called)
                mock_run.called = False
                self.assertIsNone(subject.run_script('test_script', []))
                self.assertFalse(mock_run.called)

    def test_run_script_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_run_script_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(bson.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_run_script_writes_job_script(self):
        script_name = 'demo_script_' + str(bson.ObjectId())
        script_args = ['--test', str(bson.ObjectId())]
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script(script_name, script_args)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1}".format(script_name, ' '.join(script_args)), script_contents)

    def test_run_script_indicates_desired_cpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [], num_cpus=15789, num_gpus=0)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ncpus={cpus}".format(cpus=15789), script_contents)
        self.assertNotIn("#PBS -l cputype=", script_contents)

    def test_run_script_indicates_desired_gpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [], num_gpus=8026)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ngpus={gpus}".format(gpus=8026), script_contents)
        self.assertIn("#PBS -l gputype=M40", script_contents)
        self.assertIn("#PBS -l cputype=E5-2680v4", script_contents)

    def test_run_script_indicates_desired_memory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [], memory_requirements='1542GB')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l mem={mem}".format(mem='1542GB'), script_contents)

    def test_run_script_indicates_expected_run_time(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [], expected_duration='125:23:16')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l walltime={time}".format(time='125:23:16'), script_contents)

    def test_run_script_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "auto_task_[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_script_job_name_has_configured_prefix(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "job_auto_task_[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_script_uses_configured_environment(self):
        mock_open = mock.mock_open()
        virtualenv_path = '/home/user/virtualenv/benchmark-framework/bin/activate'
        subject = hpc.HPCJobSystem({'environment': virtualenv_path})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}'.format(virtualenv_path), script_contents)

    def test_run_script_uses_virtualenv_from_environment(self):
        virtualenv_path = '/home/user/virtualenv/benchmark-framework'
        mock_open = mock.mock_open()
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.os.environ', {'VIRTUAL_ENV': virtualenv_path}):
            subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    def test_run_script_uses_current_working_directory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_script('test_script', [])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('cd {0}'.format(os.getcwd()), script_contents)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_script_task_submits_job(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.run_script('test_script', [])
        filename = mock_open.call_args[0][0]
        self.assertIn(mock.call(['qsub', filename], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    def test_run_script_returns_job_id(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run, job_id=25798)
                result = subject.run_script('test_script', [])
        self.assertEqual(25798, result)

    def test_run_task_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_run_task_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(bson.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_run_task_writes_job_script(self):
        mock_task = make_mock_task()
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(mock_task)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1}".format(
            hpc.quote(arvet.batch_analysis.scripts.run_task.__file__), mock_task.identifier
        ), script_contents)

    def test_run_task_indicates_desired_cpus(self):
        mock_task = make_mock_task()
        mock_task.num_cpus = 15789

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(mock_task)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ncpus={cpus}".format(cpus=15789), script_contents)
        self.assertNotIn("#PBS -l cputype=", script_contents)

    def test_run_task_indicates_desired_gpus(self):
        mock_task = make_mock_task()
        mock_task.num_gpus = 8026

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(mock_task)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ngpus={gpus}".format(gpus=8026), script_contents)
        self.assertIn("#PBS -l gputype=M40", script_contents)
        self.assertIn("#PBS -l cputype=E5-2680v4", script_contents)

    def test_run_task_indicates_desired_memory(self):
        mock_task = make_mock_task()
        mock_task.memory_requirements = '1542GB'

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(mock_task)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l mem={mem}".format(mem='1542GB'), script_contents)

    def test_run_task_indicates_expected_run_time(self):
        mock_task = make_mock_task()
        mock_task.expected_duration = '125:23:16'

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(mock_task)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l walltime={time}".format(time='125:23:16'), script_contents)

    def test_run_task_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "auto_task_[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_task_job_name_has_configured_prefix(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "job_auto_task_[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_task_uses_configured_environment(self):
        mock_open = mock.mock_open()
        virtualenv_path = '/home/user/virtualenv/benchmark-framework/bin/activate'
        subject = hpc.HPCJobSystem({'environment': virtualenv_path})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}'.format(virtualenv_path), script_contents)

    def test_run_task_uses_virtualenv_from_environment(self):
        virtualenv_path = '/home/user/virtualenv/benchmark-framework'
        mock_open = mock.mock_open()
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.os.environ', {'VIRTUAL_ENV': virtualenv_path}):
            subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    def test_run_task_uses_current_working_directory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(make_mock_task())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('cd {0}'.format(os.getcwd()), script_contents)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_task_submits_job(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.run_task(make_mock_task())
        filename = mock_open.call_args[0][0]
        self.assertIn(mock.call(['qsub', filename], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    def test_run_task_returns_job_id(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run, job_id=25798)
                result = subject.run_task(make_mock_task())
        self.assertEqual(25798, result)

    def test_quote_passes_through_strings_without_spaces(self):
        string = 'this-is_a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual(string, hpc.quote(string))

    def test_quote_wraps_a_string_containing_spaces_in_double_quotes(self):
        string = 'this-is a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual('"' + string + '"', hpc.quote(string))


def patch_subprocess_qjobs(mock_subprocess_run: mock.Mock, num_jobs=4):
    mock_completed_process = mock.Mock()
    mock_completed_process.stdout = """

    {0} running jobs found found for test-user
    to see finished jobs, run qjobs with the -x flag


    """.format(num_jobs)
    mock_subprocess_run.return_value = mock_completed_process


def patch_subprocess(mock_subprocess_run: mock.Mock, job_id=4, num_jobs=15):
    mock_qjobs_completed_process = mock.Mock()
    mock_qjobs_completed_process.stdout = """

    {0} running jobs found found for test-user
    to see finished jobs, run qjobs with the -x flag


    """.format(num_jobs)

    mock_qsub_completed_process = mock.Mock()
    mock_qsub_completed_process.stdout = str(job_id)

    mock_subprocess_run.side_effect = lambda script_args, *args, **kwargs: \
        mock_qjobs_completed_process if script_args[0] is 'qjobs' else mock_qsub_completed_process


def make_mock_task():
    mock_task = mock.create_autospec(Task)
    mock_task.identifier = bson.ObjectId()
    mock_task.num_cpus = 1
    mock_task.num_gpus = 0
    mock_task.memory_requirements = '4GB'
    mock_task.expected_duration = '1:00:00'
    return mock_task
