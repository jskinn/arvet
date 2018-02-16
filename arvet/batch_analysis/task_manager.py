# Copyright (c) 2017, John Skinner
import copy
import pymongo.collection
import bson
import typing
import arvet.util.database_helpers as dh
import arvet.util.dict_utils as du
import arvet.database.client
import arvet.database.entity_registry as entity_registry
import arvet.batch_analysis.job_system
import arvet.batch_analysis.task
import arvet.batch_analysis.tasks.import_dataset_task as import_dataset_task
import arvet.batch_analysis.tasks.generate_dataset_task as generate_dataset_task
import arvet.batch_analysis.tasks.train_system_task as train_system_task
import arvet.batch_analysis.tasks.run_system_task as run_system_task
import arvet.batch_analysis.tasks.benchmark_trial_task as benchmark_task
import arvet.batch_analysis.tasks.compare_trials_task as compare_trials_task
import arvet.batch_analysis.tasks.compare_benchmarks_task as compare_benchmarks_task
import arvet.batch_analysis.scripts.warmup_image_cache


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

    def __init__(self, task_collection: pymongo.collection.Collection,
                 db_client: arvet.database.client.DatabaseClient, config: dict = None):
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

    def get_import_dataset_task(self, module_name: str, path: str, additional_args: dict = None,
                                num_cpus: int =1, num_gpus: int = 0,
                                memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
        """
        Get a task to import a dataset.
        Most of the parameters are resources requirements passed to the job system.
        :param module_name: The name of the python module to use to do the import as a string.
        It must have a function 'import_dataset', taking a directory and the database client
        :param path: The root file or directory describing the dataset to import
        :param additional_args: Additional arguments to the importer module, depends on what module is chosen
        :param num_cpus: The number of CPUs required for the job. Default 1.
        :param num_gpus: The number of GPUs required for the job. Default 0.
        :param memory_requirements: The memory required for this job. Default 3 GB.
        :param expected_duration: The expected time this job will take. Default 1 hour.
        :return: An ImportDatasetTask containing the task state.
        """
        if additional_args is None:
            additional_args = {}
        existing = self._collection.find_one(dh.query_to_dot_notation({
            'module_name': module_name, 'path': path,
            'additional_args': copy.deepcopy(additional_args)}))
        if existing is not None:
            return self._db_client.deserialize_entity(existing)
        else:
            return import_dataset_task.ImportDatasetTask(
                module_name=module_name,
                path=path,
                additional_args=additional_args,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                memory_requirements=memory_requirements,
                expected_duration=expected_duration
            )

    def get_generate_dataset_task(self, controller_id: bson.ObjectId, simulator_id: bson.ObjectId,
                                  simulator_config: dict, repeat: int = 0,
                                  num_cpus: int = 1, num_gpus: int = 0,
                                  memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
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

    def get_train_system_task(self, trainer_id: bson.ObjectId, trainee_id: bson.ObjectId,
                              num_cpus: int = 1, num_gpus: int = 0,
                              memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
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

    def get_run_system_task(self, system_id: bson.ObjectId, image_source_id: bson.ObjectId, repeat: int = 0,
                            num_cpus: int = 1, num_gpus: int = 0,
                            memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
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

    def get_benchmark_task(self, trial_result_id: bson.ObjectId, benchmark_id: bson.ObjectId,
                           num_cpus: int = 1, num_gpus: int = 0,
                           memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
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

    def get_trial_comparison_task(self, trial_result1_id: bson.ObjectId, trial_result2_id: bson.ObjectId,
                                  comparison_id: bson.ObjectId, num_cpus: int = 1, num_gpus: int = 0,
                                  memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
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

    def get_benchmark_comparison_task(self, benchmark_result1_id: bson.ObjectId, benchmark_result2_id: bson.ObjectId,
                                      comparison_id: bson.ObjectId, num_cpus: int = 1, num_gpus: int = 0,
                                      memory_requirements: str = '3GB', expected_duration: str = '1:00:00'):
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

    def do_task(self, task: arvet.batch_analysis.task.Task):
        """
        Submit a task back to the task manager for execution.
        Simply getting the task is not enough, you need to pass it here for the task to be run.

        Will check if the task already exists
        :param task: The task object to run, must be an instance of Task returned by the above get methods
        :return: void
        """
        if isinstance(task, arvet.batch_analysis.task.Task) and task.identifier is None and task.is_unstarted:
            existing_query = {}
            # Each different task type has a different set of properties that identify it.
            if isinstance(task, import_dataset_task.ImportDatasetTask):
                existing_query['module_name'] = task.module_name
                existing_query['path'] = task.path
                existing_query['additional_args'] = task.additional_args
                dh.query_to_dot_notation(existing_query)
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
                existing_query['repeat'] = task.repeat
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

    def schedule_tasks(self, job_system: arvet.batch_analysis.job_system.JobSystem):
        """
        Schedule all pending tasks using the provided job system
        This should both star
        :param job_system:
        :return:
        """
        # First, check the jobs that should already be running on this node
        all_running = self._collection.find({
            'state': arvet.batch_analysis.task.JobState.RUNNING.value,
            'node_id': job_system.node_id
        })
        for s_running in all_running:
            task_entity = self._db_client.deserialize_entity(s_running)
            if not job_system.is_job_running(task_entity.job_id):
                # Task should be running, but job system says it isn't re-run
                task_entity.mark_job_failed()
                task_entity.save_updates(self._collection)

        # Then, schedule all the unscheduled tasks that we are configured to allow.
        # Start with everything other than the run system tasks, they get special handling
        types = []
        if self._allow_import_dataset:
            types.append(entity_registry.get_type_name(import_dataset_task.ImportDatasetTask))
        if self._allow_generate_dataset:
            types.append(entity_registry.get_type_name(generate_dataset_task.GenerateDatasetTask))
        if self._allow_train_system:
            types.append(entity_registry.get_type_name(train_system_task.TrainSystemTask))
        if self._allow_benchmark:
            types.append(entity_registry.get_type_name(benchmark_task.BenchmarkTrialTask))
        if self._allow_trial_comparison:
            types.append(entity_registry.get_type_name(compare_trials_task.CompareTrialTask))
        if self._allow_benchmark_comparison:
            types.append(entity_registry.get_type_name(compare_benchmarks_task.CompareBenchmarksTask))
        if len(types) > 0:
            for s_task in self._collection.find({
                'state': arvet.batch_analysis.task.JobState.UNSTARTED.value,
                '_type': {'$in': types}
            }):
                task_entity = self._db_client.deserialize_entity(s_task)
                job_id = job_system.run_task(
                    task_id=task_entity.identifier,
                    num_cpus=task_entity.num_cpus,
                    num_gpus=task_entity.num_gpus,
                    memory_requirements=task_entity.memory_requirements,
                    expected_duration=task_entity.expected_duration
                )
                task_entity.mark_job_started(job_system.node_id, job_id)
                task_entity.save_updates(self._collection)

        # Then, we want to group up run system tasks together by image source, so that we can use the cache
        # group all the run system tasks
        if self._allow_run_system:
            run_groups = {}
            for s_task in self._collection.find({
                'state': arvet.batch_analysis.task.JobState.UNSTARTED.value,
                '_type': entity_registry.get_type_name(run_system_task.RunSystemTask)
            }):
                task_entity = self._db_client.deserialize_entity(s_task)
                if task_entity.image_source not in run_groups:
                    run_groups[task_entity.image_source] = [task_entity]
                else:
                    run_groups[task_entity.image_source].append(task_entity)

            # Finally, schedule warmup cache scripts, with each list of tasks as dependent jobs
            for image_source_id, task_list in run_groups.items():
                job_id = job_system.run_script(
                    script=arvet.batch_analysis.scripts.warmup_image_cache.__file__,
                    script_args=['--image_collection', str(image_source_id)] + [str(t.identifier) for t in task_list],
                    num_cpus=1,
                    num_gpus=0,
                    memory_requirements='12GB',
                    expected_duration='12:00:00'
                )
                for task_entity in task_list:
                    task_entity.mark_job_started(job_system.node_id, job_id)
                    task_entity.save_updates(self._collection)

    def schedule_dependent_tasks(self, task_ids: typing.List[bson.ObjectId],
                                 job_system: arvet.batch_analysis.job_system.JobSystem):
        """
        Schedule a list of tasks that depended on some other job that we have now completed.
        Only those tasks allowed by this job system will be scheduled.
        For book-keeping, we change the job_id of the tasks to their newly-submitted job id
        :param task_ids:
        :param job_system:
        :return:
        """
        types = []
        if self._allow_import_dataset:
            types.append(entity_registry.get_type_name(import_dataset_task.ImportDatasetTask))
        if self._allow_generate_dataset:
            types.append(entity_registry.get_type_name(generate_dataset_task.GenerateDatasetTask))
        if self._allow_train_system:
            types.append(entity_registry.get_type_name(train_system_task.TrainSystemTask))
        if self._allow_run_system:
            types.append(entity_registry.get_type_name(run_system_task.RunSystemTask))
        if self._allow_benchmark:
            types.append(entity_registry.get_type_name(benchmark_task.BenchmarkTrialTask))
        if self._allow_trial_comparison:
            types.append(entity_registry.get_type_name(compare_trials_task.CompareTrialTask))
        if self._allow_benchmark_comparison:
            types.append(entity_registry.get_type_name(compare_benchmarks_task.CompareBenchmarksTask))
        if len(types) > 0:
            for s_task in self._collection.find({'_id': {'$in': task_ids}, '_type': {'$in': types}}):
                task_entity = self._db_client.deserialize_entity(s_task)
                job_id = job_system.run_task(
                    task_id=task_entity.identifier,
                    num_cpus=task_entity.num_cpus,
                    num_gpus=task_entity.num_gpus,
                    memory_requirements=task_entity.memory_requirements,
                    expected_duration=task_entity.expected_duration
                )
                task_entity.change_job_id(job_system.node_id, job_id)
                task_entity.save_updates(self._collection)
