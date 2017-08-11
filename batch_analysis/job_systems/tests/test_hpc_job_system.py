import unittest
import unittest.mock as mock
import os
import bson.objectid as oid
import batch_analysis.job_systems.hpc_job_system as hpc
import task_import_dataset
import task_train_system
import task_run_system
import task_benchmark_result


class TestHPCJobSystem(unittest.TestCase):

    def test_works_with_empty_config(self):
        hpc.HPCJobSystem({})

    def test_cannot_generate_dataset(self):
        subject = hpc.HPCJobSystem({})
        self.assertFalse(subject.can_generate_dataset(oid.ObjectId(), {}))

    def test_queue_generate_dataset_returns_false(self):
        subject = hpc.HPCJobSystem({})
        self.assertFalse(subject.queue_generate_dataset(oid.ObjectId(), {}))

    def test_queue_import_dataset_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertNotIn('/tmp/dataset', filename)  # we shouldn't add arbitrary slashes
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_queue_import_dataset_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(oid.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_queue_import_dataset_writes_job_script(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1} {2}".format(
            hpc.quote(task_import_dataset.__file__), 'dataset.importer', '/tmp/dataset'), script_contents)

    def test_queue_import_dataset_indicates_desired_cpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset', num_cpus=15789, num_gpus=0)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ncpus={cpus}".format(cpus=15789), script_contents)
        self.assertNotIn("#PBS -l cputype=", script_contents)

    def test_queue_import_dataset_indicates_desired_gpus(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset', num_gpus=8026)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l ngpus={gpus}".format(gpus=8026), script_contents)
        self.assertIn("#PBS -l gputype=M40", script_contents)
        self.assertIn("#PBS -l cputype=E5-2680v4", script_contents)

    def test_queue_import_dataset_indicates_desired_memory(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset', memory_requirements='1542GB')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l mem={mem}".format(mem='1542GB'), script_contents)

    def test_queue_import_dataset_indicates_expected_run_time(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset', expected_duration='125:23:16')
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn("#PBS -l walltime={time}".format(time='125:23:16'), script_contents)

    def test_queue_import_dataset_passes_experiment_to_task(self):
        mock_open = mock.mock_open()
        experiment_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset', experiment_id)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn(
            "python {0} {1} {2} {3}".format(
                hpc.quote(task_import_dataset.__file__),
                'dataset.importer',
                '/tmp/dataset',
                str(experiment_id)),
            script_contents)

    def test_queue_import_dataset_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "import-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_import_dataset_job_name_has_configured_prefix(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "job_import-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_import_dataset_uses_configured_environment(self):
        mock_open = mock.mock_open()
        virtualenv_path = '/home/user/virtualenv/benchmark-framework/bin/activate'
        subject = hpc.HPCJobSystem({'environment': virtualenv_path})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
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
            subject.queue_import_dataset('dataset.importer', '/tmp/dataset')
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    def test_queue_import_dataset_returns_true(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            self.assertTrue(subject.queue_import_dataset('dataset.importer', '/tmp/dataset'))

    def test_queue_train_system_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        trainer_id = oid.ObjectId()
        trainee_id = oid.ObjectId()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(trainer_id, trainee_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_queue_train_system_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(oid.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(oid.ObjectId(), oid.ObjectId())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_queue_train_system_writes_job_script(self):
        mock_open = mock.mock_open()
        trainer_id = oid.ObjectId()
        trainee_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(trainer_id, trainee_id)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1} {2}".format(
            hpc.quote(task_train_system.__file__), str(trainer_id), str(trainee_id)), script_contents)

    def test_queue_train_system_passes_experiment_to_task(self):
        mock_open = mock.mock_open()
        trainer_id = oid.ObjectId()
        trainee_id = oid.ObjectId()
        experiment_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(trainer_id, trainee_id, experiment_id)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn(
            "python {0} {1} {2} {3}".format(
                hpc.quote(task_train_system.__file__),
                str(trainer_id),
                str(trainee_id),
                str(experiment_id)),
            script_contents)

    def test_queue_train_system_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        trainer_id = oid.ObjectId()
        trainee_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(trainer_id, trainee_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "train-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_train_system_job_name_has_configured_prefix(self):
        mock_open = mock.mock_open()
        trainer_id = oid.ObjectId()
        trainee_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(trainer_id, trainee_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "job_train-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_train_system_uses_configured_virtualenv(self):
        mock_open = mock.mock_open()
        virtualenv_path = '/home/user/virtualenv/benchmark-framework/bin/activate'
        subject = hpc.HPCJobSystem({'environment': virtualenv_path})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(oid.ObjectId(), oid.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}'.format(virtualenv_path), script_contents)

    def test_queue_train_system_uses_virtualenv_from_environment(self):
        virtualenv_path = '/home/user/virtualenv/benchmark-framework'
        mock_open = mock.mock_open()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.os.environ', {'VIRTUAL_ENV': virtualenv_path}):
            subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_train_system(oid.ObjectId(), oid.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    def test_queue_train_system_returns_true(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            self.assertTrue(subject.queue_train_system(oid.ObjectId(), oid.ObjectId()))

    def test_queue_run_system_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(system_id, image_source_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_queue_run_system_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(oid.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(oid.ObjectId(), oid.ObjectId())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_queue_run_system_writes_job_script(self):
        mock_open = mock.mock_open()
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(system_id, image_source_id)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1} {2}".format(
            hpc.quote(task_run_system.__file__), str(system_id), str(image_source_id)), script_contents)

    def test_queue_run_system_passes_experiment_to_task(self):
        mock_open = mock.mock_open()
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        experiment_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(system_id, image_source_id, experiment_id)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn(
            "python {0} {1} {2} {3}".format(
                hpc.quote(task_run_system.__file__),
                str(system_id),
                str(image_source_id),
                str(experiment_id)),
            script_contents)

    def test_queue_run_system_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(system_id, image_source_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "run-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_run_system_job_name_has_configured_prefix(self):
        mock_open = mock.mock_open()
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(system_id, image_source_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "job_run-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_run_system_uses_configured_virtualenv(self):
        mock_open = mock.mock_open()
        virtualenv_path = '/home/user/virtualenv/benchmark-framework/bin/activate'
        subject = hpc.HPCJobSystem({'environment': virtualenv_path})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(oid.ObjectId(), oid.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}'.format(virtualenv_path), script_contents)

    def test_queue_run_system_uses_virtualenv_from_environment(self):
        virtualenv_path = '/home/user/virtualenv/benchmark-framework'
        mock_open = mock.mock_open()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.os.environ', {'VIRTUAL_ENV': virtualenv_path}):
            subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_run_system(oid.ObjectId(), oid.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    def test_queue_run_system_returns_true(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            self.assertTrue(subject.queue_run_system(oid.ObjectId(), oid.ObjectId()))

    def test_queue_benchmark_result_creates_job_file(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        subject = hpc.HPCJobSystem({})
        trial_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(trial_id, benchmark_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(os.path.expanduser('~')),
                        "{0} is not in the home directory".format(filename))
        self.assertTrue(filename.endswith('.sub'), "{0} does not end with '.sub'".format(filename))

    def test_queue_benchmark_result_creates_job_file_in_configured_directory(self):
        mock_open = mock.mock_open()
        mock_open.return_value = mock.MagicMock()
        target_folder = os.path.join('/tmp', 'trial-{}'.format(oid.ObjectId()))
        subject = hpc.HPCJobSystem({'job_location': target_folder})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(oid.ObjectId(), oid.ObjectId())
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        # Creates in the home directory by default
        self.assertTrue(filename.startswith(target_folder),
                        "{0} is not in the target directory".format(filename))

    def test_queue_benchmark_result_writes_job_script(self):
        mock_open = mock.mock_open()
        trial_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(trial_id, benchmark_id)
        self.assertTrue(mock_open.called)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertTrue(script_contents.startswith('#!/bin/bash'), "Did not create a bash script")
        self.assertIn("python {0} {1} {2}".format(
            hpc.quote(task_benchmark_result.__file__), str(trial_id), str(benchmark_id)), script_contents)

    def test_queue_benchmark_result_passes_experiment_to_task(self):
        mock_open = mock.mock_open()
        trial_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        experiment_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(trial_id, benchmark_id, experiment_id)
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn(
            "python {0} {1} {2} {3}".format(
                hpc.quote(task_benchmark_result.__file__),
                str(trial_id),
                str(benchmark_id),
                str(experiment_id)),
            script_contents)

    def test_queue_benchmark_result_job_name_matches_filename(self):
        mock_open = mock.mock_open()
        trial_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(trial_id, benchmark_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "benchmark-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_benchmark_result_job_name_has_configured_prefix(self):
        mock_open = mock.mock_open()
        trial_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()

        subject = hpc.HPCJobSystem({'job_name_prefix': 'job_'})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(trial_id, benchmark_id)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertRegex(filename, "job_benchmark-[0-9-]+\.sub$")
        _, _, filename = filename.rpartition('/')
        filename, _, _ = filename.rpartition('.')
        self.assertNotEqual('', filename)
        # Check that the name in the job file is exactly the same as the script name
        self.assertIn('#PBS -N {0}'.format(filename), script_contents)

    def test_queue_benchmark_result_uses_configured_environment(self):
        mock_open = mock.mock_open()
        virtualenv_path = '/home/user/virtualenv/benchmark-framework/bin/activate'
        subject = hpc.HPCJobSystem({'environment': virtualenv_path})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(oid.ObjectId(), oid.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}'.format(virtualenv_path), script_contents)

    def test_queue_benchmark_result_uses_virtualenv_from_environment(self):
        virtualenv_path = '/home/user/virtualenv/benchmark-framework'
        mock_open = mock.mock_open()
        with mock.patch('batch_analysis.job_systems.hpc_job_system.os.environ', {'VIRTUAL_ENV': virtualenv_path}):
            subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            subject.queue_benchmark_result(oid.ObjectId(), oid.ObjectId())
        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)
        script_contents = mock_file.write.call_args[0][0]
        self.assertIn('source {0}/bin/activate'.format(virtualenv_path), script_contents)

    def test_queue_benchmark_result_returns_true(self):
        mock_open = mock.mock_open()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock_open, create=True):
            self.assertTrue(subject.queue_benchmark_result(oid.ObjectId(), oid.ObjectId()))

    @mock.patch('batch_analysis.job_systems.hpc_job_system.time')
    @mock.patch('batch_analysis.job_systems.hpc_job_system.subprocess')
    def test_push_queued_jobs_runs_qsub_with_each_job_file(self, mock_subprocess, mock_time):
        mock_time.time = mock.MagicMock()
        mock_time.time.return_value = 123456789
        trainer_id = oid.ObjectId()
        trainee_id = oid.ObjectId()
        system_id = oid.ObjectId()
        image_source_id = oid.ObjectId()
        trial_id = oid.ObjectId()
        benchmark_id = oid.ObjectId()
        subject = hpc.HPCJobSystem({})
        with mock.patch('batch_analysis.job_systems.hpc_job_system.open', mock.mock_open(), create=True):
            subject.queue_train_system(trainer_id, trainee_id)
            subject.queue_run_system(system_id, image_source_id)
            subject.queue_benchmark_result(trial_id, benchmark_id)
        self.assertFalse(mock_subprocess.call.called)
        subject.push_queued_jobs()
        self.assertEqual(3, mock_subprocess.call.call_count)
        self.assertIn(mock.call(['qsub', os.path.expanduser("~/train-123456789.sub")]),
                      mock_subprocess.call.call_args_list)
        self.assertIn(mock.call(['qsub', os.path.expanduser("~/run-123456789.sub")]),
                      mock_subprocess.call.call_args_list)
        self.assertIn(mock.call(['qsub', os.path.expanduser("~/benchmark-123456789.sub")]),
                      mock_subprocess.call.call_args_list)

    def test_quote_passes_through_strings_without_spaces(self):
        string = 'this-is_a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual(string, hpc.quote(string))

    def test_quote_wraps_a_string_containing_spaces_in_double_quotes(self):
        string = 'this-is a#string!@#$%^&**)12344575{0}},./'
        self.assertEqual('"' + string + '"', hpc.quote(string))
