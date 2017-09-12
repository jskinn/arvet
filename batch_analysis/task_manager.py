#Copyright (c) 2017, John Skinner
import copy
import util.database_helpers as dh
import util.dict_utils as du
import batch_analysis.task
import batch_analysis.tasks.import_dataset_task as import_dataset_task
import batch_analysis.tasks.generate_dataset_task as generate_dataset_task
import batch_analysis.tasks.train_system_task as train_system_task
import batch_analysis.tasks.run_system_task as run_system_task
import batch_analysis.tasks.benchmark_trial_task as benchmark_task
import batch_analysis.tasks.compare_trials_task as compare_trials_task
import batch_analysis.tasks.compare_benchmarks_task as compare_benchmarks_task


class TaskManager:
    """
    The task manager's job is to track what has been done and what hasn't, and see that it happens.
    This means that any experiment should be able to ask the TaskManager for a task,
    and get the result if it has alredy been done. If it hasn't, the TaskManager should use the job system
    to schedule the desired task.
    If the job system fails to complete a task for whatever reason, it is the task manager's job to retry the task.

    Most of the interface is done through Task objects, which are handles back to particular executions.
    We guarantee exactly 3 things about a Task object:
    - 'is_finished' indicates whether the task has finished running, and the value in 'result' is valid
    - 'result' contains the result of the run, or None if it isn't finished yet. Usually this is an id.
    - You can pass a task object back to the task manager to 'queue_task' which will cause an unfinished task to be run.
    Everything else in the task behaviour is an implementation detail of TaskManager, and shouldn't be relied on.
    """

    def __init__(self, task_collection, db_client, config=None):
        """
        Create the task manager, wrapping a collection of tasks
        :param task_collection: The collection containing the tasks
        :param db_client: The database client, for deserialization
        """
        self._collection = task_collection
        self._db_client = db_client
        self._pending_tasks = []

        # configuration keys, to avoid misspellings
        if config is not None and 'task_config' in config:
            task_config = dict(config['task_config'])
        else:
            task_config = {}

        # Default configuration. Also serves as an exemplar configuration argument
        du.defaults(task_config, {
            'allow_generate_dataset': True,
            'allow_import_dataset': True,
            'allow_train_system': True,
            'allow_run_system': True,
            'allow_benchmark': True,
            'allow_trial_comparison': True,
            'allow_benchmark_comparison': True
        })
        self._allow_generate_dataset = bool(task_config['allow_generate_dataset'])
        self._allow_import_dataset = bool(task_config['allow_import_dataset'])
        self._allow_train_system = bool(task_config['allow_train_system'])
        self._allow_run_system = bool(task_config['allow_run_system'])
        self._allow_benchmark = bool(task_config['allow_benchmark'])
        self._allow_trial_comparison = bool(task_config['allow_trial_comparison'])
        self._allow_benchmark_comparison = bool(task_config['allow_benchmark_comparison'])

    def get_import_dataset_task(self, module_name, path, num_cpus=1, num_gpus=0,
                                memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Get a task to import a dataset.
        Most of the parameters are resources requirements passed to the job system.
        :param module_name: The name of the python module to use to do the import as a string.
        It must have a function 'import_dataset', taking a directory and the database client
        :param path: The root file or directory describing the dataset to import
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: An ImportDatasetTask containing the task state.
        """
        existing = self._collection.find_one({'module_name': module_name, 'path': path})
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return import_dataset_task.ImportDatasetTask(
                module_name=module_name,
                path=path,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_generate_dataset_task(self, controller_id, simulator_id, simulator_config, repeat=0, num_cpus=1, num_gpus=0,
                                  memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Get a task to generate a synthetic dataset.
        Generate dataset tasks are unique to particular combinations of controller, simulator and config,
        so that the same controller can generate different datasets with the same simulator.
        This is further enabled by the repeat parameter.
        Most of the parameters are resources requirements passed to the job system.
        :param controller_id: The id of the controller to use
        :param simulator_id: The id of the simulator to use
        :param simulator_config: configuration parameters passed to the simulator at run time.
        :param repeat: The repeat of this trial, so we can run the same system more than once.
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: An ImportDatasetTask containing the task state.
        """
        existing = self._collection.find_one(dh.query_to_dot_notation({
            'controller_id': controller_id, 'simulator_id': simulator_id,
            'simulator_config': copy.deepcopy(simulator_config), 'repeat': repeat}))
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return generate_dataset_task.GenerateDatasetTask(
                controller_id=controller_id,
                simulator_id=simulator_id,
                simulator_config=simulator_config,
                repeat=repeat,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_train_system_task(self, trainer_id, trainee_id, num_cpus=1, num_gpus=0,
                              memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Create a task to train a system.
        Most of the parameters are resources requirements passed to the job system.
        :param trainer_id: The id of the trainer to do the training
        :param trainee_id: The id of the trainee to train
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: A TrainSystemTask
        """
        existing = self._collection.find_one({'trainer_id': trainer_id, 'trainee_id': trainee_id})
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return train_system_task.TrainSystemTask(
                trainer_id=trainer_id,
                trainee_id=trainee_id,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_run_system_task(self, system_id, image_source_id, repeat=0, num_cpus=1, num_gpus=0,
                            memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Get a task to run a system.
        Most of the parameters are resources requirements passed to the job system.
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param repeat: The repeat of this trial, so we can run the same system more than once.
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: A RunSystemTask
        """
        existing = self._collection.find_one({
            'system_id': system_id,
            'image_source_id': image_source_id,
            'repeat': repeat
        })
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return run_system_task.RunSystemTask(
                system_id=system_id,
                image_source_id=image_source_id,
                repeat=repeat,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_benchmark_task(self, trial_result_id, benchmark_id, num_cpus=1, num_gpus=0,
                           memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Get a task to benchmark a trial result.
        Most of the parameters are resources requirements passed to the job system.
        :param trial_result_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: A BenchmarkTrialTask
        """
        existing = self._collection.find_one({'trial_result_id': trial_result_id, 'benchmark_id': benchmark_id})
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return benchmark_task.BenchmarkTrialTask(
                trial_result_id=trial_result_id,
                benchmark_id=benchmark_id,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_trial_comparison_task(self, trial_result1_id, trial_result2_id, comparison_id, num_cpus=1, num_gpus=0,
                                  memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Get a task to compare two trial results.
        Most of the parameters are resources requirements passed to the job system.
        :param trial_result1_id: The id of the first trial result to compare
        :param trial_result2_id: The id of the second trial result to compare
        :param comparison_id: The id of the comparison benchmark to use
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: A BenchmarkTrialTask
        """
        existing = self._collection.find_one({'trial_result1_id': trial_result1_id,
                                              'trial_result2_id': trial_result2_id, 'comparison_id': comparison_id})
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return compare_trials_task.CompareTrialTask(
                trial_result1_id=trial_result1_id,
                trial_result2_id=trial_result2_id,
                comparison_id=comparison_id,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_benchmark_comparison_task(self, benchmark_result1_id, benchmark_result2_id, comparison_id,
                                      num_cpus=1, num_gpus=0, memory_requirements='3GB', expected_duration='1:00:00'):
        """
        Get a task to compare two benchmark results.
        Most of the parameters are resources requirements passed to the job system.
        :param benchmark_result1_id: The id of the first benchmark result to compare
        :param benchmark_result2_id: The id of the second benchmark result to compare
        :param comparison_id: The id of the benchmark to perform the comparison
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: A BenchmarkTrialTask
        """
        existing = self._collection.find_one({'benchmark_result1_id': benchmark_result1_id,
                                              'benchmark_result2_id': benchmark_result2_id,
                                              'comparison_id': comparison_id})
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return compare_benchmarks_task.CompareBenchmarksTask(
                benchmark_result1_id=benchmark_result1_id,
                benchmark_result2_id=benchmark_result2_id,
                comparison_id=comparison_id,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def do_task(self, task):
        """
        Submit a task back to the task manager for execution.
        Simply getting the task is not enough, you need to pass it here for the task to be run.

        Will check if the task already exists
        :param task: The task object to run, must be an instance of Task returned by the above get methods
        :return: void
        """
        if isinstance(task, batch_analysis.task.Task) and task.identifier is None and task.is_unstarted:
            existing_query = {}
            # Each different task type has a different set of properties that identify it.
            if isinstance(task, import_dataset_task.ImportDatasetTask):
                existing_query['module_name'] = task.module_name
                existing_query['path'] = task.path
            elif isinstance(task, generate_dataset_task.GenerateDatasetTask):
                existing_query['controller_id'] = task.controller_id
                existing_query['simulator_id'] = task.simulator_id
                existing_query['simulator_config'] = copy.deepcopy(task.simulator_config)
                existing_query['repeat'] = task.repeat
                existing_query = dh.query_to_dot_notation(existing_query)
            elif isinstance(task, train_system_task.TrainSystemTask):
                existing_query['trainer_id'] = task.trainer
                existing_query['trainee_id'] = task.trainee
            elif isinstance(task, run_system_task.RunSystemTask):
                existing_query['system_id'] = task.system
                existing_query['image_source_id'] = task.image_source
            elif isinstance(task, benchmark_task.BenchmarkTrialTask):
                existing_query['trial_result_id'] = task.trial_result
                existing_query['benchmark_id'] = task.benchmark
            elif isinstance(task, compare_trials_task.CompareTrialTask):
                existing_query['trial_result1_id'] = task.trial_result1
                existing_query['trial_result2_id'] = task.trial_result2
                existing_query['comparison_id'] = task.comparison
            elif isinstance(task, compare_benchmarks_task.CompareBenchmarksTask):
                existing_query['benchmark_result1_id'] = task.benchmark_result1
                existing_query['benchmark_result2_id'] = task.benchmark_result2
                existing_query['comparison_id'] = task.comparison

            # Make sure none of this task already exists
            if existing_query != {} and self._collection.find(existing_query).limit(1).count() == 0:
                task.save_updates(self._collection)

    def schedule_tasks(self, job_system):
        """
        Schedule all pending tasks using the provided job system
        This should both star
        :param job_system:
        :return:
        """
        # First, check the jobs that should already be running on this node
        all_running = self._collection.find({
            'state': batch_analysis.task.JobState.RUNNING.value,
            'node_id': job_system.node_id
        })
        for s_running in all_running:
            task_entity = self._db_client.deserialize_entity(s_running)
            if not job_system.is_job_running(task_entity.job_id):
                # Task should be running, but job system says it isn't re-run
                task_entity.mark_job_failed()
                task_entity.save_updates(self._collection)

        # Then, schedule all the unscheduled tasks that we are configured to allow
        all_unscheduled = self._collection.find({'state': batch_analysis.task.JobState.UNSTARTED.value})
        for s_unscheduled in all_unscheduled:
            task_entity = self._db_client.deserialize_entity(s_unscheduled)
            if ((isinstance(task_entity, import_dataset_task.ImportDatasetTask) and self._allow_import_dataset) or
                    (isinstance(task_entity, generate_dataset_task.GenerateDatasetTask) and self._allow_generate_dataset) or
                    (isinstance(task_entity, train_system_task.TrainSystemTask) and self._allow_train_system) or
                    (isinstance(task_entity, run_system_task.RunSystemTask) and self._allow_run_system) or
                    (isinstance(task_entity, benchmark_task.BenchmarkTrialTask) and self._allow_benchmark) or
                    (isinstance(task_entity, compare_trials_task.CompareTrialTask) and self._allow_trial_comparison) or
                    (isinstance(task_entity, compare_benchmarks_task.CompareBenchmarksTask) and self._allow_benchmark_comparison)):
                job_id = job_system.run_task(
                    task_id=task_entity.identifier,
                    num_cpus=task_entity.num_cpus,
                    num_gpus=task_entity.num_gpus,
                    memory_requirements=task_entity.memory_requirements,
                    expected_duration=task_entity.expected_duration
                )
                task_entity.mark_job_started(job_system.node_id, job_id)
                task_entity.save_updates(self._collection)
