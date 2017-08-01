import batch_analysis.job_system
import task_import_dataset
import task_train_system
import task_run_system
import task_benchmark_result


class SimpleJobSystem(batch_analysis.job_system.JobSystem):
    """
    The worst possible, and simplest, job system.
    Simply does the job as part of scheduling it.
    No multiprocess, nothing, just direct execution.
    Still implements a job queueing system,
    so that we can defer the execution of jobs until we've finished creating them.
    """

    def __init__(self, config):
        self._queue = []

    def can_generate_dataset(self, simulator, config):
        """
        Can this job system generate synthetic datasets?
        This job system can.
        :param simulator: The simulator id that will be doing the generation
        :param config: Configuration passed to the simulator at run time
        :return: True
        """
        return True

    def queue_generate_dataset(self, simulator_id, config, experiment=None):
        """
        Queue generating a synthetic dataset using a particular simulator and a particular configuration
        :param simulator_id: The id of the simulator to use to generate the dataset
        :param config: Configuration passed to the simulator to control the dataset generation
        :param experiment: The experiment this generated dataset is associated with, if any
        :return: True iff the job was successfully queued
        """
        if experiment is not None:
            self._queue.append()
        else:
            self._queue.append()

    def queue_import_dataset(self, module_name, directory, experiment=None):
        """
        Use the job system to import a dataset
        :param module_name: The name of the python module to use to do the import as a string.
        It must have a function 'import_dataset', taking a directory and the database client
        :param directory: The root directory containing the dataset to import
        :param experiment: The experiment to give the imported dataset to, if any
        :return: True iff the job was queued
        """
        if experiment is not None:
            self._queue.append((task_import_dataset.main, (module_name, directory, experiment)))
        else:
            self._queue.append((task_import_dataset.main, (module_name, directory)))

    def queue_train_system(self, trainer_id, trainee_id, experiment=None):
        """
        Train a system, now, in the current process
        :param trainer_id: The id of the trainer doing the training
        :param trainee_id: The id of the trainee being trained
        :param experiment: The experiment associated with this run, if any
        :return: void
        """
        if experiment is not None:
            self._queue.append((task_train_system.main, (trainer_id, trainee_id, experiment)))
        else:
            self._queue.append((task_train_system.main, (trainer_id, trainee_id)))

    def queue_run_system(self, system_id, image_source_id, experiment=None):
        """
        Run a system, now, in the current process
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param experiment: The experiment associated with this run, if any
        :return: void
        """
        if experiment is not None:
            self._queue.append((task_run_system.main, (system_id, image_source_id, experiment)))
        else:
            self._queue.append((task_run_system.main, (system_id, image_source_id)))

    def queue_benchmark_result(self, trial_id, benchmark_id, experiment=None):
        """
        Do a benchmark, now, in the current process.
        :param trial_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param experiment: The experiment this is associated with, if any
        :return: void
        """
        if experiment is not None:
            self._queue.append((task_benchmark_result.main, (trial_id, benchmark_id, experiment)))
        else:
            self._queue.append((task_benchmark_result.main, (trial_id, benchmark_id)))

    def push_queued_jobs(self):
        """
        Actually run the jobs.
        :return: void
        """
        for func, args in self._queue:
            func(*args)
        self._queue = []
