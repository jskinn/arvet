# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import logging
import os
import bson
from pathlib import Path
from arvet.batch_analysis.task import Task
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
from arvet.batch_analysis.tasks.run_system_task import RunSystemTask
from arvet.batch_analysis.tasks.measure_trial_task import MeasureTrialTask
from arvet.batch_analysis.tasks.compare_trials_task import CompareTrialTask
import arvet.batch_analysis.job_systems.hpc_job_system as hpc
import arvet.batch_analysis.scripts.run_task


class TestHPCJobSystem(unittest.TestCase):

    def test_works_with_empty_config(self):
        hpc.HPCJobSystem({}, 'myconf.yml')

    def test_cannot_generate_dataset(self):
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        self.assertFalse(subject.can_generate_dataset(bson.ObjectId(), {}))

    def test_configure_node_id(self):
        node_id = 'test-node-id-157890'
        subject = hpc.HPCJobSystem({'node_id': node_id}, 'myconf.yml')
        self.assertEqual(node_id, subject.node_id)

    def test_parse_memory_requirements_returns_num_in_KB(self):
        self.assertEqual(34 * 1024 * 1024, hpc.parse_memory_requirements('34GB'))
        self.assertEqual(23 * 1024, hpc.parse_memory_requirements('23MB'))
        self.assertEqual(196, hpc.parse_memory_requirements('196KB'))
        self.assertEqual(2554, hpc.parse_memory_requirements('2554'))

    def test_merge_expected_durations_sums_times(self):
        self.assertEqual('04:14:06', hpc.merge_expected_durations(['1:0:1', '0:12:2', '3:2:3']))

    def test_merge_expected_durations_rolls_over_seconds(self):
        self.assertEqual('00:08:48', hpc.merge_expected_durations(['0:0:43', '0:5:28', '0:2:37']))
        self.assertEqual('03:03:08', hpc.merge_expected_durations(['0:10:4313', '0:5:2238', '0:2:3417']))

    def test_merge_expected_durations_rolls_over_minutes(self):
        self.assertEqual('01:41:48', hpc.merge_expected_durations(['0:22:03', '0:50:18', '0:29:27']))
        self.assertEqual('13:54:35', hpc.merge_expected_durations(['0:102:06', '0:509:13', '0:223:16']))

    def test_quote_passes_through_strings_without_spaces(self):
        string = 'this-is_a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual(string, hpc.quote(string))

    def test_quote_wraps_a_string_containing_spaces_in_double_quotes(self):
        string = 'this-is a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual('"' + string + '"', hpc.quote(string))


class TestHPCJobSystemRunScript(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_script_checks_number_of_existing_jobs_if_a_limit_is_set(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({'max_jobs': 100}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt:
            patch_import_dataset_count(mock_idt)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()   # Doesn't actually make the job until we run the queue
        self.assertIn(mock.call(['qjobs'], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_script_does_not_check_number_of_existing_jobs_if_no_limit_is_set(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt:
            patch_import_dataset_count(mock_idt)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        self.assertNotIn(mock.call(['qjobs'], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    def test_run_script_wont_submit_more_scripts_than_job_limit(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'max_jobs': 10}, 'myconf.yml')
        for idx in range(12):
            subject.run_script('test_script_{0}'.format(idx), lambda *_: [])

        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run, num_jobs=0)
            subject.run_queued_jobs()
            self.assertEqual(11, mock_run.call_count)   # Plus 1 for the check for existing
            # Make sure there are exactly 10 qsub calls
            self.assertEqual(10, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))

    def test_existing_jobs_reduce_max_jobs(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'max_jobs': 10}, 'myconf.yml')
        for idx in range(12):
            subject.run_script('test_script_{0}'.format(idx), lambda *_: [])

        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run, num_jobs=5)
            subject.run_queued_jobs()
            self.assertEqual(6, mock_run.call_count)  # Plus 1 for the check for existing
            # Make sure there are exactly 5 qsub calls
            self.assertEqual(5, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))

    def test_run_script_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        # Creates in the home directory by default
        self.assertTrue(str(filename).startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_run_script_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(bson.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(str(filename).startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_run_script_writes_job_script(self):
        script_name = 'demo_script_' + str(bson.ObjectId())
        script_args = ['--test', str(bson.ObjectId())]
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script(script_name, lambda *_: script_args)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1}".format(script_name, ' '.join(script_args)), script_contents)

    def test_run_script_indicates_desired_cpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [], num_cpus=15789, num_gpus=0)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ncpus={cpus}".format(cpus=15789), script_contents)
        self.assertNotIn("#PBS -l cputype=", script_contents)

    def test_run_script_indicates_desired_gpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [], num_gpus=8026)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ngpus={gpus}".format(gpus=8026), script_contents)
        self.assertIn("#PBS -l gputype=M40", script_contents)
        self.assertIn("#PBS -l cputype=E5-2680v4", script_contents)

    def test_run_script_indicates_desired_memory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [], memory_requirements='1542GB')
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l mem={mem}".format(mem='1542GB'), script_contents)

    def test_run_script_indicates_expected_run_time(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [], expected_duration='125:23:16')
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l walltime={time}".format(time='125:23:16'), script_contents)

    def test_run_script_job_name_matches_filename(self):
        job_name = 'my_job'
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [], job_name=job_name)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(filename.endswith("{0}.sub".format(job_name)))
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertEqual(job_name, filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_script_job_name_has_configured_prefix(self):
        job_name = 'myjob'
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [], job_name=job_name)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(filename.endswith("job_{0}.sub".format(job_name)))
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_script_creates_job_file_that_runs_script_with_returned_arguments(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda config, port=0: ['--myarg1', str(config)])
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('\npython test_script --myarg1 {0}\n'.format(os.path.abspath('myconf.yml')), script_contents)

    def test_run_script_uses_configured_environment(self):
        mock_open = mock.mock_open()
        environment_path = '/home/user/environment.sh'
        subject = hpc.HPCJobSystem({'environment': environment_path}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        # split the script into what happens before the script is called, and what happens after
        parts = script_contents.split('\npython test_script')
        self.assertEqual(2, len(parts))
        self.assertIn('\nsource {0}'.format(environment_path), parts[0])

    def test_run_script_uses_current_working_directory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        # split the script into what happens before the script is called, and what happens after
        parts = script_contents.split('\npython test_script')
        self.assertEqual(2, len(parts))
        self.assertIn('\ncd {0}'.format(os.getcwd()), parts[0])

    def test_run_script_removes_job_after_complete(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        job_filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        # split the script into what happens before the script is called, and what happens after
        parts = script_contents.split('\npython test_script')
        self.assertEqual(2, len(parts))
        self.assertIn('\nrm '.format(job_filename), parts[1])

    def test_run_script_doesnt_do_anything_with_ssh_if_not_configured_to(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertNotIn('ssh', script_contents)

    def test_run_script_starts_and_stops_ssh_tunnel(self):
        job_name = 'testjob'
        mock_open = mock.mock_open()
        environment_path = '/home/user/environment.sh'
        job_folder = str(Path(__file__).parent)
        subject = hpc.HPCJobSystem({
            'job_location': job_folder,
            'environment': environment_path,
            'ssh_tunnel': {
                'hostname': '127.0.0.1',
                'username': 'test-user',
                'ssh_key': 'my-ssh-key.rsa',
                'min_port': 5060,
                'max_port': 5062
            }
        }, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run, num_jobs=0)
            subject.run_script('test_script', lambda *_: [], job_name=job_name)
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        # split the script into what happens before the script is called, and what happens after
        parts = script_contents.split('\npython test_script')
        self.assertEqual(2, len(parts))
        self.assertIn(hpc.SSH_TUNNEL_PREFIX.format(
            ssh_key=os.path.abspath('my-ssh-key.rsa'),
            local_port=5060,
            username='test-user',
            hostname='127.0.0.1',
            job_name=job_name,
            job_folder=job_folder
        ).strip(), parts[0])

        self.assertIn(hpc.SSH_TUNNEL_SUFFIX.format(
            local_port=5060,
            job_name=job_name,
            job_folder=job_folder
        ).strip(), parts[1])

    def test_run_script_assigns_different_ssh_ports_to_successive_jobs(self):
        job_name = 'testjob'
        min_port = 5060
        mock_open = mock.mock_open()
        environment_path = '/home/user/environment.sh'
        job_folder = str(Path(__file__).parent)
        subject = hpc.HPCJobSystem({
            'job_location': job_folder,
            'environment': environment_path,
            'ssh_tunnel': {
                'hostname': '127.0.0.1',
                'username': 'test-user',
                'ssh_key': 'my-ssh-key.rsa',
                'min_port': min_port,
                'max_port': min_port + 1000
            }
        }, 'myconf.yml')

        for port in range(min_port, min_port + 10):
            with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                 mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                    mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
                patch_import_dataset_count(mock_idt)
                patch_subprocess(mock_run)
                subject.run_script('test_script', lambda config, p=0: ['--port', str(p)], job_name=job_name)
                subject.run_queued_jobs()
            mock_file = mock_open()
            self.assertTrue(mock_file.write.called)
            script_contents = mock_file.write.call_args[0][0]
            self.assertIn('\npython test_script --port {0}\n'.format(port), script_contents)

            # split the script into what happens before the script is called, and what happens after
            parts = script_contents.split('\npython test_script')
            self.assertEqual(2, len(parts))
            self.assertIn(hpc.SSH_TUNNEL_PREFIX.format(
                ssh_key=os.path.abspath('my-ssh-key.rsa'),
                local_port=port,
                username='test-user',
                hostname='127.0.0.1',
                job_name=job_name,
                job_folder=job_folder
            ).strip(), parts[0])

            self.assertIn(hpc.SSH_TUNNEL_SUFFIX.format(
                local_port=port,
                job_name=job_name,
                job_folder=job_folder
            ).strip(), parts[1])

    def test_run_script_wont_submit_more_scripts_than_available_ssh_ports(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({
            'environment': '/home/user/environment.sh',
            'ssh_tunnel': {
                'hostname': '127.0.0.1',
                'username': 'test-user',
                'ssh_key': 'my-ssh-key.rsa',
                'min_port': 5060,
                'max_port': 5069    # Max is inclusive
            }
        }, 'myconf.yml')
        for _ in range(12):
            subject.run_script('test_script', lambda *_: [])
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_count(mock_idt)
            patch_subprocess(mock_run, num_jobs=0)
            subject.run_queued_jobs()
            self.assertEqual(11, mock_run.call_count)  # Plus 1 for the check for existing
            # Make sure there are exactly 5 qsub calls
            self.assertEqual(10, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_script_task_submits_job(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask') as mock_idt:
            patch_import_dataset_count(mock_idt)
            subject.run_script('test_script', lambda *_: [])
            subject.run_queued_jobs()
        filename = mock_open.call_args[0][0]
        self.assertIn(mock.call(['qsub', filename], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)


class TestHPCJobSystemRunTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_run_task_marks_task_started(self):
        node_id = 'node-14170'
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        mock_task = make_mock_task()
        subject = hpc.HPCJobSystem({'node_id': node_id}, 'myconf.yml')
        subject.run_task(mock_task)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()

        self.assertTrue(mock_task.mark_job_started.called)
        self.assertEqual(node_id, mock_task.mark_job_started.call_args[0][0])
        self.assertTrue(mock_task.save.called)

    def test_run_task_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        subject.run_task(make_mock_task())
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()

        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_run_task_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(bson.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(make_mock_task())
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_run_task_writes_job_script(self):
        mock_task = make_mock_task()
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} --config {1} {2}".format(
            hpc.quote(arvet.batch_analysis.scripts.run_task.__file__),
            os.path.abspath('myconf.yml'),
            mock_task.pk
        ), script_contents)

    def test_run_task_indicates_desired_cpus(self):
        mock_task = make_mock_task()
        mock_task.num_cpus = 15789

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
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
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
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
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l mem={mem}".format(mem='1542GB'), script_contents)

    def test_run_task_indicates_expected_run_time(self):
        mock_task = make_mock_task()
        mock_task.expected_duration = '125:23:16'

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l walltime={time}".format(time='125:23:16'), script_contents)

    def test_run_task_job_name_matches_filename(self):
        mock_task = make_mock_task()
        job_name = mock_task.get_unique_name()
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(filename.endswith("{0}.sub".format(job_name)))
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_task_job_name_has_configured_prefix(self):
        mock_task = make_mock_task()
        job_name = mock_task.get_unique_name()
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(mock_task)
            subject.run_queued_jobs()
        self.assertTrue(mock_open.called)
        filename = str(mock_open.call_args[0][0])
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(filename.endswith("job_{0}.sub".format(job_name)))
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_run_task_creates_job_file_that_calls_run_task(self):
        mock_open = mock.mock_open()
        mock_task = make_mock_task()
        environment_path = '/home/user/environment.sh'
        subject = hpc.HPCJobSystem({'environment': environment_path}, 'myconf.yml')
        subject.run_task(mock_task)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('\npython {0} --config {1} {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(), mock_task.pk
        ), script_contents)

    def test_run_task_uses_configured_environment(self):
        mock_open = mock.mock_open()
        environment_path = '/home/user/environment.sh'
        subject = hpc.HPCJobSystem({'environment': environment_path}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(make_mock_task())
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        # split the script into what happens before the script is called, and what happens after
        parts = script_contents.split('\npython')
        self.assertEqual(2, len(parts))
        self.assertIn('\nsource {0}'.format(environment_path), parts[0])

    def test_run_task_uses_current_working_directory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(make_mock_task())
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        # split the script into what happens before the script is called, and what happens after
        parts = script_contents.split('\npython')
        self.assertEqual(2, len(parts))
        self.assertIn('\ncd {0}'.format(os.getcwd()), parts[0])

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_task_submits_job(self, mock_run):
        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(make_mock_task())
            subject.run_queued_jobs()
        filename = mock_open.call_args[0][0]
        self.assertIn(mock.call(['qsub', filename], stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)

    def test_run_task_wont_run_import_dataset_task_if_configured_not_to(self):
        task = ImportDatasetTask(_id=bson.ObjectId())

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'task_config': {'allow_import_dataset': False}}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(task)
            subject.run_queued_jobs()
            self.assertFalse(mock_run.called)

    def test_run_task_wont_run_system_if_configured_not_to(self):
        task = RunSystemTask(_id=bson.ObjectId())

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'task_config': {'allow_run_system': False}}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(task)
            self.assertFalse(mock_run.called)

    def test_run_task_wont_measure_trials_if_configured_not_to(self):
        task = MeasureTrialTask(_id=bson.ObjectId())

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'task_config': {'allow_run_system': False}}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(task)
            self.assertFalse(mock_run.called)

    def test_run_task_wont_compare_trials_if_configured_not_to(self):
        task = CompareTrialTask(_id=bson.ObjectId())

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'task_config': {'allow_run_system': False}}, 'myconf.yml')
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_task(task)
            self.assertFalse(mock_run.called)

    def test_run_task_makes_single_script_for_all_import_dataset_tasks(self):
        normal_task = make_mock_task()
        import_task_1 = make_mock_import_task()
        import_task_2 = make_mock_import_task()
        import_task_3 = make_mock_import_task()

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        subject.run_task(normal_task)
        subject.run_task(import_task_1)
        subject.run_task(import_task_2)
        subject.run_task(import_task_3)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()
        self.assertEqual(1, mock_open.call_count)
        self.assertEqual(1, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))

        mock_file = mock_open()
        self.assertEqual(1, mock_file.write.call_count)
        script_contents = mock_file.write.call_args[0][0]

        self.assertTrue(import_task_1.mark_job_started.called)
        self.assertTrue(import_task_2.mark_job_started.called)
        self.assertTrue(import_task_3.mark_job_started.called)
        self.assertIn('\npython {0} --config {1} --allow_write {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(),
            import_task_1.pk
        ), script_contents)
        self.assertIn('\npython {0} --config {1} --allow_write {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(),
            import_task_2.pk
        ), script_contents)
        self.assertIn('\npython {0} --config {1} --allow_write {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(),
            import_task_3.pk
        ), script_contents)

    def test_run_task_accepts_only_a_limited_number_of_import_dataset_tasks(self):
        normal_task = make_mock_task()
        import_task_1 = make_mock_import_task()
        import_task_2 = make_mock_import_task()
        import_task_3 = make_mock_import_task()

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'max_imports': 2}, 'myconf.yml')
        subject.run_task(normal_task)
        subject.run_task(import_task_1)
        subject.run_task(import_task_2)
        subject.run_task(import_task_3)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()
        self.assertEqual(1, mock_open.call_count)
        self.assertEqual(1, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))

        mock_file = mock_open()
        self.assertEqual(1, mock_file.write.call_count)
        script_contents = mock_file.write.call_args[0][0]

        # The first two tasks should appear in the file, but the third should not.
        self.assertTrue(import_task_1.mark_job_started.called)
        self.assertTrue(import_task_2.mark_job_started.called)
        self.assertFalse(import_task_3.mark_job_started.called)
        self.assertIn('\npython {0} --config {1} --allow_write {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(),
            import_task_1.pk
        ), script_contents)
        self.assertIn('\npython {0} --config {1} --allow_write {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(),
            import_task_2.pk
        ), script_contents)
        self.assertNotIn('\npython {0} --config {1} --allow_write {2}\n'.format(
            Path(arvet.batch_analysis.scripts.run_task.__file__).resolve(),
            Path('myconf.yml').resolve(),
            import_task_3.pk
        ), script_contents)

    def test_run_queued_jobs_doesnt_run_other_tasks_if_there_are_imports_to_run(self):
        normal_task = make_mock_task()
        import_task_1 = make_mock_import_task()
        import_task_2 = make_mock_import_task()
        import_task_3 = make_mock_import_task()

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        subject.run_task(normal_task)
        subject.run_task(import_task_1)
        subject.run_task(import_task_2)
        subject.run_task(import_task_3)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()
        self.assertEqual(1, mock_open.call_count)
        self.assertEqual(1, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))

        mock_file = mock_open()
        self.assertEqual(1, mock_file.write.call_count)
        script_contents = mock_file.write.call_args[0][0]
        self.assertNotIn(str(normal_task.pk), script_contents)
        self.assertFalse(normal_task.mark_job_started.called)

    def test_run_task_merges_import_dataset_requirements(self):
        import_task_1 = make_mock_import_task(num_cpus=1, num_gpus=3, memory=4, hours=3, minutes=43, seconds=32)
        import_task_2 = make_mock_import_task(num_cpus=3, num_gpus=0, memory=8, hours=1, minutes=55, seconds=45)
        import_task_3 = make_mock_import_task(num_cpus=5, num_gpus=1, memory=2, hours=2, minutes=36, seconds=59)

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        subject.run_task(import_task_1)
        subject.run_task(import_task_2)
        subject.run_task(import_task_3)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]

        self.assertIn("\n#PBS -l ncpus={cpus}".format(cpus=5), script_contents)
        self.assertIn("\n#PBS -l ngpus={gpus}".format(gpus=3), script_contents)
        self.assertIn("\n#PBS -l mem={mem}".format(mem='8GB'), script_contents)
        self.assertIn("\n#PBS -l walltime={time}".format(time='08:16:16'), script_contents)

    def test_run_task_marks_all_import_dataset_tasks_as_started(self):
        node_id = 'node-2308'
        import_task_1 = make_mock_import_task()
        import_task_2 = make_mock_import_task()
        import_task_3 = make_mock_import_task()

        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'node_id': node_id}, 'myconf.yml')
        subject.run_task(import_task_1)
        subject.run_task(import_task_2)
        subject.run_task(import_task_3)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj, \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run') as mock_run:
            patch_import_dataset_objects_count(mock_idt_obj)
            patch_subprocess(mock_run)
            subject.run_queued_jobs()
        self.assertTrue(import_task_1.mark_job_started.called)
        self.assertEqual(node_id, import_task_1.mark_job_started.call_args[0][0])
        job_id = import_task_1.mark_job_started.call_args[0][1]
        self.assertTrue(import_task_2.mark_job_started.called)
        self.assertEqual(mock.call(node_id, job_id), import_task_2.mark_job_started.call_args)
        self.assertTrue(import_task_3.mark_job_started.called)
        self.assertEqual(mock.call(node_id, job_id), import_task_3.mark_job_started.call_args)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_queued_jobs_doesnt_submit_any_tasks_if_imports_are_already_running(self, mock_run):
        normal_task = make_mock_task()
        import_task = make_mock_import_task()

        mock_open = mock.mock_open()
        patch_subprocess(mock_run)
        subject = hpc.HPCJobSystem({}, 'myconf.yml')
        subject.run_task(normal_task)
        subject.run_task(import_task)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj:
            patch_import_dataset_objects_count(mock_idt_obj, num_tasks=1)
            subject.run_queued_jobs()
        self.assertEqual(0, mock_open.call_count)
        self.assertEqual(0, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))
        self.assertFalse(normal_task.mark_job_started.called)
        self.assertFalse(import_task.mark_job_started.called)

        mock_file = mock_open()
        self.assertFalse(mock_file.write.called)

    @mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.subprocess.run')
    def test_run_queued_jobs_submits_least_failed_jobs_first(self, mock_run):
        task_1 = make_mock_task()
        task_1.failure_count = 10
        task_2 = make_mock_task()
        task_2.failure_count = 0
        task_3 = make_mock_task()
        task_3.failure_count = 4

        mock_open = mock.mock_open()
        patch_subprocess(mock_run, num_jobs=0)
        subject = hpc.HPCJobSystem({'max_jobs': 2}, 'myconf.yml')
        subject.run_task(task_1)
        subject.run_task(task_2)
        subject.run_task(task_3)
        with mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True), \
                mock.patch('arvet.batch_analysis.job_systems.hpc_job_system.ImportDatasetTask.objects',
                           spec=ImportDatasetTask.objects) as mock_idt_obj:
            patch_import_dataset_objects_count(mock_idt_obj)
            subject.run_queued_jobs()
        self.assertEqual(2, sum(1 for call in mock_run.call_args_list if call[0][0][0] == 'qsub'))
        self.assertIn(mock.call(['qsub', subject._job_folder / (task_2.get_unique_name() + '.sub')],
                                stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)
        self.assertIn(mock.call(['qsub', subject._job_folder / (task_3.get_unique_name() + '.sub')],
                                stdout=mock.ANY, universal_newlines=True), mock_run.call_args_list)
        self.assertFalse(task_1.mark_job_started.called)
        self.assertTrue(task_2.mark_job_started.called)
        self.assertTrue(task_3.mark_job_started.called)


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


def patch_import_dataset_count(mock_import_dataset, num_tasks=0):
    patch_import_dataset_objects_count(mock_import_dataset.objects, num_tasks)


def patch_import_dataset_objects_count(mock_import_dataset_objects, num_tasks=0):
    mock_cursor = mock.Mock()
    mock_cursor.count.return_value = num_tasks
    mock_import_dataset_objects.raw.return_value = mock_cursor


def make_mock_task():
    mock_task = mock.create_autospec(Task)
    mock_task.pk = bson.ObjectId()
    mock_task.num_cpus = 1
    mock_task.num_gpus = 0
    mock_task.memory_requirements = '4GB'
    mock_task.expected_duration = '1:00:00'
    mock_task.get_unique_name.return_value = 'job_' + str(mock_task.pk)
    return mock_task


def make_mock_import_task(num_cpus=1, num_gpus=0, memory=4, hours=1, minutes=0, seconds=0):
    mock_task = mock.create_autospec(ImportDatasetTask)
    mock_task.pk = bson.ObjectId()
    mock_task.num_cpus = num_cpus
    mock_task.num_gpus = num_gpus
    mock_task.memory_requirements = '{0}GB'.format(memory)
    mock_task.expected_duration = "{0:02}:{1:02}:{2:02}".format(hours, minutes, seconds)
    mock_task.get_unique_name.return_value = 'job_' + str(mock_task.pk)
    return mock_task
