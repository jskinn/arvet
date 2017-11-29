# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os
import bson
import batch_analysis.job_systems.hpc_job_system as hpc
import run_task


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

    def test_run_task_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId())
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
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_run_task_writes_job_script(self):
        task_id = bson.ObjectId()
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(task_id)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1}".format(hpc.quote(run_task.__file__), task_id), script_contents)

    def test_run_task_indicates_desired_cpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId(), num_cpus=15789, num_gpus=0)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ncpus={cpus}".format(cpus=15789), script_contents)
        self.assertNotIn("#PBS -l cputype=", script_contents)

    def test_run_task_indicates_desired_gpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId(), num_gpus=8026)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ngpus={gpus}".format(gpus=8026), script_contents)
        self.assertIn("#PBS -l gputype=M40", script_contents)
        self.assertIn("#PBS -l cputype=E5-2680v4", script_contents)

    def test_run_task_indicates_desired_memory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId(), memory_requirements='1542GB')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l mem={mem}".format(mem='1542GB'), script_contents)

    def test_run_task_indicates_expected_run_time(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId(), expected_duration='125:23:16')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l walltime={time}".format(time='125:23:16'), script_contents)

    def test_queue_import_dataset_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId())
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
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId())
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
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}'.format(virtualenv_path), script_contents)

    def test_queue_import_dataset_uses_virtualenv_from_environment(self):
        virtualenv_path = '/home/user/virtualenv/benchmark-framework'
        mock_open = mock.mock_open()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.os.environ', {'VIRTUAL_ENV': virtualenv_path}):
            subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run)
                subject.run_task(bson.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    @mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_task_submits_job(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.run_task(bson.ObjectId())
        filename = mock_open.call_args[0][0]
        self.assertIn(mock.call(['qsub', filename], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    def test_run_task_returns_job_id(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            with mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_subprocess(mock_run, job_id=25798)
                result = subject.run_task(bson.ObjectId())
        self.assertEqual(25798, result)

    def test_quote_passes_through_strings_without_spaces(self):
        string = 'this-is_a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual(string, hpc.quote(string))

    def test_quote_wraps_a_string_containing_spaces_in_double_quotes(self):
        string = 'this-is a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual('"' + string + '"', hpc.quote(string))


def patch_subprocess(mock_subprocess_run: mock.Mock, job_id=4):
    mock_completed_process = mock.Mock()
    mock_completed_process.stdout = str(job_id)
    mock_subprocess_run.return_value = mock_completed_process
