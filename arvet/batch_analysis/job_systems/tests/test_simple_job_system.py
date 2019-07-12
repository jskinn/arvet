# Copyright (c) 2019, John Skinner
import unittest
import unittest.mock as mock
import bson
from arvet.batch_analysis.task import Task
import arvet.batch_analysis.scripts.run_task
from arvet.batch_analysis.job_systems.simple_job_system import SimpleJobSystem


class TestSimpleJobSystem(unittest.TestCase):

    def test_works_with_empty_config(self):
        SimpleJobSystem({})

    def test_can_generate_dataset(self):
        subject = SimpleJobSystem({})
        self.assertTrue(subject.can_generate_dataset(bson.ObjectId(), {}))

    def test_configure_node_id(self):
        node_id = 'test-node-id-157890'
        subject = SimpleJobSystem({'node_id': node_id})
        self.assertEqual(node_id, subject.node_id)

    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.subprocess.run')
    def test_run_script_adds_script_to_queue(self, mock_run):
        subject = SimpleJobSystem({})
        subject.run_script('myscript.py', ['--myarg'])
        self.assertFalse(mock_run.called)
        subject.run_queued_jobs()
        self.assertTrue(mock_run.called)

        run_args = mock_run.call_args[0][0]
        self.assertIn('python', run_args)
        python_index = run_args.index('python')
        self.assertEqual(python_index + 3, len(run_args))
        self.assertIn('myscript.py', run_args[python_index + 1])
        self.assertIn('--myarg', run_args[python_index + 2])

    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.subprocess.run')
    def test_run_task_adds_script_to_queue(self, mock_run):
        task = make_mock_task()
        subject = SimpleJobSystem({})
        subject.run_task(task)
        self.assertFalse(mock_run.called)
        subject.run_queued_jobs()
        self.assertTrue(mock_run.called)

        run_args = mock_run.call_args[0][0]
        self.assertIn('python', run_args)
        python_index = run_args.index('python')
        self.assertEqual(python_index + 3, len(run_args))
        self.assertEqual(arvet.batch_analysis.scripts.run_task.__file__, run_args[python_index + 1])
        self.assertEqual(str(task.identifier), run_args[python_index + 2])

    def test_is_job_running_returns_true_for_jobids_returned_from_run_script(self):
        subject = SimpleJobSystem({})
        job_id = subject.run_script('myscript.py', ['--myarg'])
        self.assertFalse(subject.is_job_running(-1))
        self.assertTrue(subject.is_job_running(job_id))
        self.assertFalse(subject.is_job_running(1))
        self.assertFalse(subject.is_job_running(10))

    def test_is_job_running_returns_true_for_jobids_returned_from_run_task(self):
        task = make_mock_task()
        subject = SimpleJobSystem({})
        job_id = subject.run_task(task)
        self.assertFalse(subject.is_job_running(-1))
        self.assertTrue(subject.is_job_running(job_id))
        self.assertFalse(subject.is_job_running(1))
        self.assertFalse(subject.is_job_running(10))

    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.os')
    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.subprocess.run')
    def test_run_queued_jobs_finds_and_uses_a_conda_environment(self, mock_run, mock_os):
        # Create the environment variable used to indicate the presence of the conda env
        mock_os.environ = {'CONDA_DEFAULT_ENV': 'my_conda_env'}

        subject = SimpleJobSystem({})
        subject.run_script('myscript.py', ['--myarg'])
        subject.run_queued_jobs()
        self.assertTrue(mock_run.called)
        run_args = mock_run.call_args[0][0]
        self.assertEqual(['conda', 'run', '-n', 'my_conda_env', 'python', 'myscript.py', '--myarg'], run_args)

    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.os')
    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.subprocess.run')
    def test_run_queued_jobs_uses_default_python_by_default(self, mock_run, mock_os):
        # Mock the environ to hide any conda/virtualenv environment running the test
        mock_os.environ = {}
        subject = SimpleJobSystem({})
        subject.run_script('myscript.py', ['--myarg'])
        subject.run_queued_jobs()
        self.assertTrue(mock_run.called)
        run_args = mock_run.call_args[0][0]
        self.assertEqual(['python', 'myscript.py', '--myarg'], run_args)

    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.os')
    @mock.patch('arvet.batch_analysis.job_systems.simple_job_system.subprocess.run')
    def test_run_queued_jobs_passes_through_cwd(self, mock_run, mock_os):
        # Mock the environ to hide any conda/virtualenv environment running the test
        mock_os.getcwd.return_value = '/my/current/working/directory'
        subject = SimpleJobSystem({})
        subject.run_script('myscript.py', ['--myarg'])
        subject.run_queued_jobs()
        self.assertTrue(mock_run.called)
        run_kwargs = mock_run.call_args[1]
        self.assertEqual({'cwd': '/my/current/working/directory'}, run_kwargs)


def make_mock_task():
    mock_task = mock.create_autospec(Task)
    mock_task.identifier = bson.ObjectId()
    mock_task.num_cpus = 1
    mock_task.num_gpus = 0
    mock_task.memory_requirements = '4GB'
    mock_task.expected_duration = '1:00:00'
    return mock_task
