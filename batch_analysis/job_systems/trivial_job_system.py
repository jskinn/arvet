import batch_analysis.job_system
import task_run_system
import task_benchmark_result


class TrivialJobSystem(batch_analysis.job_system.JobSystem):
    """
    The worst possible, and simplest, job system.
    Simply does the job as part of scheduling it.
    No multiprocess, nothing, just direct execution.
    """

    def schedule_run_system(self, system_id, image_source_id, experiment=None):
        """
        Run a system, now, in the current process
        :param system_id: The id of the vision system to test
        :param image_source_id: The id of the image source to test with
        :param experiment: The experiment associated with this run, if any
        :return: void
        """
        if experiment is not None:
            task_run_system.main(system_id, image_source_id, experiment)
        else:
            task_run_system.main(system_id, image_source_id)

    def schedule_benchmark_result(self, trial_id, benchmark_id, experiment=None):
        """
        Do a benchmark, now, in the current process.
        :param trial_id: The id of the trial result to benchmark
        :param benchmark_id: The id of the benchmark to use
        :param experiment: The experiment this is associated with, if any
        :return: void
        """
        if experiment is not None:
            task_benchmark_result.main(trial_id, benchmark_id, experiment)
        else:
            task_benchmark_result.main(trial_id, benchmark_id)
