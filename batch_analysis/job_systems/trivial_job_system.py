import batch_analysis.job_system
import task_run_system
import task_benchmark_result


class TrivialJobSystem(batch_analysis.job_system.JobSystem):
    """
    The worst possible, and simplest, job system.
    Simply does the job as part of scheduling it.
    No multiprocess, nothing, just direct execution.
    Still implements a job queueing system,
    so that we can defer the execution of jobs until we've finished creating them.
    """

    def __init__(self):
        self._queue = []

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
