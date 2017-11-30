import unittest.mock as mock
import argus.batch_analysis.task_manager
import argus.batch_analysis.tasks.import_dataset_task as idt
import argus.batch_analysis.tasks.generate_dataset_task as gdt
import argus.batch_analysis.tasks.train_system_task as tst
import argus.batch_analysis.tasks.run_system_task as rst
import argus.batch_analysis.tasks.benchmark_trial_task as btt
import argus.batch_analysis.tasks.compare_trials_task as ctt
import argus.batch_analysis.tasks.compare_benchmarks_task as cbt


class ZombieTaskManager:
    """
    A wrapper for a mock task manager that hooks into it's mock methods, so that they actually work.
    The object is still a MagicMock, and can be inspected for all the normal properties .
    This means that the task manager can be injected in, and will provide the correct output as the code expects,
    while

    All the state is held externally in this object, to avoid polluting the TaskManager interface and retain
    the benefits of using a spec.
    """

    def __init__(self, mock_task_manager):
        self._import_dataset_tasks = {}
        self._generate_dataset_tasks = {}
        self._train_system_tasks = {}
        self._run_system_tasks = {}
        self._benchmark_tasks = {}
        self._compare_trials_tasks = {}
        self._compare_benchmarks_tasks = {}
        self._tasks_to_do = []

        # Set up the mock object to call the methods on this object as their side effect
        # I'm using some lamba context binding to link back to the zombie object, and filter out extra arguments.
        mock_task_manager.get_import_dataset_task.side_effect = (
            lambda module_name, path, *_, **__:
                self.get_import_dataset_task(module_name, path))
        mock_task_manager.get_generate_dataset_task.side_effect = (
            lambda controller_id, simulator_id, simulator_config, *_, **__:
                self.get_generate_dataset_task(controller_id, simulator_id, simulator_config))
        mock_task_manager.get_train_system_task.side_effect = (
            lambda trainer_id, trainee_id, *_, **__:
                self.get_train_system_task(trainer_id, trainee_id))
        mock_task_manager.get_run_system_task.side_effect = (
            lambda system_id, image_source_id, *_, **__:
                self.get_run_system_task(system_id, image_source_id))
        mock_task_manager.get_benchmark_task.side_effect = (
            lambda trial_result_id, benchmark_id, *_, **__:
                self.get_benchmark_task(trial_result_id, benchmark_id))
        mock_task_manager.get_trial_comparison_task.side_effect = (
            lambda trial_result1_id, trial_result2_id, comparison_id, *_, **__:
                self.get_trial_comparison_task(trial_result1_id, trial_result2_id, comparison_id))
        mock_task_manager.get_benchmark_comparison_task.side_effect = (
            lambda benchmark_result1_id, benchmark_result2_id, comparison_id, *_, **__:
                self.get_benchmark_comparison_task(benchmark_result1_id, benchmark_result2_id, comparison_id))
        self._mock_task_manager = mock_task_manager

    @property
    def mock(self) -> argus.batch_analysis.task_manager.TaskManager:
        """
        Get the mock task manager this is attached to
        :return:
        """
        return self._mock_task_manager

    def get_all_tasks(self):
        """
        Get all the tasks this object is holding, in no particular order.
        You cant use the various get methods for this, because they will create the task if it is missing.
        :return:
        """
        return ([task for inner in self._import_dataset_tasks.values() for task in inner.values()] +
                [task for inner in self._generate_dataset_tasks.values() for task in inner.values()] +
                [task for inner in self._train_system_tasks.values() for task in inner.values()] +
                [task for inner in self._run_system_tasks.values() for task in inner.values()] +
                [task for inner in self._benchmark_tasks.values() for task in inner.values()] +
                [task for inner1 in self._compare_trials_tasks.values() for inner2 in inner1.values()
                 for task in inner2.values()] +
                [task for inner1 in self._compare_benchmarks_tasks.values() for inner2 in inner1.values()
                 for task in inner2.values()])

    def add_import_dataset_task(self, task: idt.ImportDatasetTask):
        """
        Add an import dataset task. Use this to set up the initial state of the task manager
        :param task: The import dataset task to add. Should have all it's parameters set.
        :return: void
        """
        if task.module_name not in self._import_dataset_tasks:
            self._import_dataset_tasks[task.module_name] = {}
        self._import_dataset_tasks[task.module_name][task.path] = task

    def get_import_dataset_task(self, module_name, path):
        """
        Get a task to import a dataset. This is called by the mock object.
        :param module_name: The name of the python module to use to do the import as a string.
        :param path: The root file or directory describing the dataset to import
        :return: An ImportDatasetTask containing the task state.
        """
        if module_name not in self._import_dataset_tasks:
            self._import_dataset_tasks[module_name] = {}
        if path not in self._import_dataset_tasks[module_name]:
            self._import_dataset_tasks[module_name][path] = idt.ImportDatasetTask(
                module_name=module_name,
                path=path,
            )
        return self._import_dataset_tasks[module_name][path]

    def get_generate_dataset_task(self, controller_id, simulator_id, simulator_config, *_, **__):
        """
        Get a task to generate a synthetic dataset.
        :param controller_id: The id of the controller to use
        :param simulator_id: The id of the simulator to use
        :param simulator_config: configuration parameters passed to the simulator at run time.
        :return: An ImportDatasetTask containing the task state.
        """
        if controller_id not in self._generate_dataset_tasks:
            self._generate_dataset_tasks[controller_id] = {}
        if simulator_id not in self._generate_dataset_tasks[controller_id]:
            self._generate_dataset_tasks[controller_id][simulator_id] = gdt.GenerateDatasetTask(
                controller_id=controller_id,
                simulator_id=simulator_id,
                simulator_config=simulator_config
            )
        return self._generate_dataset_tasks[simulator_id][controller_id]

    def get_train_system_task(self, trainer_id, trainee_id):
        """
        Create a task to train a system.
        Most of the parameters are resources requirements passed to the job system.
        :param trainer_id: The id of the trainer to do the training
        :param trainee_id: The id of the trainee to train
        :return: A TrainSystemTask
        """
        if trainer_id not in self._train_system_tasks:
            self._train_system_tasks[trainer_id] = {}
        if trainee_id not in self._train_system_tasks[trainer_id]:
            self._train_system_tasks[trainer_id][trainee_id] = tst.TrainSystemTask(
                trainer_id=trainer_id,
                trainee_id=trainee_id
            )
        return self._train_system_tasks[trainer_id][trainee_id]

    def get_run_system_task(self, system_id, image_source_id):
        """
        Get a task to run a system.
        Most of the parameters are resources requirements passed to the job system.
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :return: A RunSystemTask
        """
        if system_id not in self._run_system_tasks:
            self._run_system_tasks[system_id] = {}
        if image_source_id not in self._run_system_tasks[system_id]:
            self._run_system_tasks[system_id][image_source_id] = rst.RunSystemTask(
                system_id=system_id,
                image_source_id=image_source_id
            )
        return self._run_system_tasks[system_id][image_source_id]

    def get_benchmark_task(self, trial_result_id, benchmark_id):
        """
        Get a task to benchmark a trial result.
        Most of the parameters are resources requirements passed to the job system.
        :param trial_result_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :return: A BenchmarkTrialTask
        """
        if trial_result_id not in self._benchmark_tasks:
            self._benchmark_tasks[trial_result_id] = {}
        if benchmark_id not in self._benchmark_tasks[trial_result_id]:
            self._benchmark_tasks[trial_result_id][benchmark_id] = btt.BenchmarkTrialTask(
                trial_result_id=trial_result_id,
                benchmark_id=benchmark_id
            )
        return self._benchmark_tasks[trial_result_id][benchmark_id]

    def get_trial_comparison_task(self, trial_result1_id, trial_result2_id, comparison_id):
        """
        Get a task to compare two trial results.
        Most of the parameters are resources requirements passed to the job system.
        :param trial_result1_id: The id of the first trial result to compare
        :param trial_result2_id: The id of the second trial result to compare
        :param comparison_id: The comparison benchmark to use
        :return: A CompareTrialTask
        """
        if comparison_id not in self._compare_trials_tasks:
            self._compare_trials_tasks[comparison_id] = {}
        if trial_result1_id not in self._compare_trials_tasks[comparison_id]:
            self._compare_trials_tasks[comparison_id][trial_result1_id] = {}
        if trial_result2_id not in self._compare_trials_tasks[comparison_id][trial_result1_id]:
            self._compare_trials_tasks[comparison_id][trial_result1_id][trial_result2_id] = ctt.CompareTrialTask(
                trial_result1_id=trial_result1_id,
                trial_result2_id=trial_result2_id,
                comparison_id=comparison_id
            )
        return self._compare_trials_tasks[comparison_id][trial_result1_id][trial_result2_id]

    def get_benchmark_comparison_task(self, benchmark_result1_id, benchmark_result2_id, comparison_id):
        """
        Get a task to compare two benchmark results.
        Most of the parameters are resources requirements passed to the job system.
        :param benchmark_result1_id: The id of the first benchmark result to compare
        :param benchmark_result2_id: The id of the second benchmark result to compare
        :param comparison_id: The id of the benchmark to perform the comparison
        :return: A CompareBenchmarksTask
        """
        if comparison_id not in self._compare_benchmarks_tasks:
            self._compare_benchmarks_tasks[comparison_id] = {}
        if benchmark_result1_id not in self._compare_benchmarks_tasks[comparison_id]:
            self._compare_benchmarks_tasks[comparison_id][benchmark_result1_id] = {}
        if benchmark_result2_id not in self._compare_benchmarks_tasks[comparison_id][benchmark_result1_id]:
            self._compare_benchmarks_tasks[comparison_id][benchmark_result1_id][benchmark_result2_id] = \
                cbt.CompareBenchmarksTask(
                benchmark_result1_id=benchmark_result1_id,
                benchmark_result2_id=benchmark_result2_id,
                comparison_id=comparison_id
            )
        return self._compare_benchmarks_tasks[comparison_id][benchmark_result1_id][benchmark_result2_id]


def create() -> ZombieTaskManager:
    """
    Create a new zombie task manager.
    The zombie task manager wraps a mock task manager that is set up to actually
    behave as a real task manager, and can be injected into objects as part of a test.
    Use the zombie object itself to construct the initial state, and then inject the mock object.
    :return: A new ZombieTaskManager
    """
    return ZombieTaskManager(mock.create_autospec(argus.batch_analysis.task_manager.TaskManager))
