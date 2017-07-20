import os
import batch_analysis.job_system


# Paths to task scripts files
TRAIN_SYSTEM_SCRIPT = 'task_train_system.py'
RUN_SYSTEM_SCRIPT = 'task_run_system.py'
BENCHMARK_RESULT_SCRIPT = 'task_benchmark_result.py'


class HPCJobSystem(batch_analysis.job_system.JobSystem):

    job_template = """
    #!/bin/python3
    {0} {1}
    """

    def queue_train_system(self, trainer_id, trainee_id, experiment=None):
        """
        Use the job system to train a system with a particular image source.
        Internally calls the 'run_script' function, above, with the "task_run_system" in the root of this project
        TODO: find a better way to get the path of the script
        :param trainer_id: The id of the trainer doing the training
        :param trainee_id: The id of the trainee being trained
        :param experiment: The experiment associated with this run, if any
        :return: void
        """
        if experiment is not None:
            self.create_job(TRAIN_SYSTEM_SCRIPT, str(trainer_id), str(trainee_id), str(experiment))
        else:
            self.create_job(TRAIN_SYSTEM_SCRIPT, str(trainer_id), str(trainee_id))

    def queue_run_system(self, system_id, image_source_id, experiment=None):
        """
        Use the job system to run a system with a particular image source.
        Internally calls the 'run_script' function, above, with the "task_run_system" in the root of this project
        TODO: find a better way to get the path of the script
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param experiment: The experiment associated with this run, if any
        :return: void
        """
        if experiment is not None:
            self.create_job(RUN_SYSTEM_SCRIPT, str(system_id), str(image_source_id), str(experiment))
        else:
            self.create_job(RUN_SYSTEM_SCRIPT, str(system_id), str(image_source_id))

    def queue_benchmark_result(self, trial_id, benchmark_id, experiment=None):
        """
        Use the job system to benchmark a particular trial result.
        Uses the 'run_script' function, above
        :param trial_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param experiment: The experiment this is associated with, if any
        :return: void
        """
        if experiment is not None:
            self.create_job(BENCHMARK_RESULT_SCRIPT, str(trial_id), str(benchmark_id), str(experiment))
        else:
            self.create_job(BENCHMARK_RESULT_SCRIPT, str(trial_id), str(benchmark_id))

    def push_queued_jobs(self):
        """
        TODO: Actually start the jobs here
        :return:
        """
        pass

    def create_job(self, script_path, *args):
        """
        Create a
        TODO: Create a HPC job and add it to the queue.
        :param script_path: The path of the python file to run, with respect to the root of this project
        :param args: Arguments passed to the script on the command line as space-separated strings
        :return: void
        """
        cwd = os.getcwd()

        # Create a job file
        args = ' '.join(args)
        script_content = self.job_template.format(os.path.abspath(os.path.join(cwd, script_path)), args)
