import numpy as np
import core.benchmark
import trials.slam.tracking_state
import benchmarks.tracking.tracking_result


class LostInterval:
    """
    An interval during which the system was lost.
    This is basically just a struct storing information about the interval
    """

    def __init__(self, start_time, end_time, distance, num_frames):
        self._start_time = start_time
        self._end_time = end_time
        self._distance = distance
        self._num_frames = num_frames

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def distance(self):
        return self._distance

    @property
    def frames(self):
        return self._num_frames


class TrackingBenchmark(core.benchmark.Benchmark):
    """
    A tool for benchmarking SLAM based on the time and distance the algorithm was lost.
    This can compute the Mean Time Lost, Mean Distance Lost,
    as well as a number of other statistics like max distance lost or median time lost.
    """

    def __init__(self, initializing_is_lost=True):
        """
        Create a tracking benchmark
        :param initializing_is_lost: Does the initializing state count as lost or not
        """
        self._initializing_is_lost = initializing_is_lost

    @property
    def identifier(self):
        return 'TrackingStatistics'

    def get_settings(self):
        return {
            'initializing_is_lost': self._initializing_is_lost
        }

    def is_lost(self, state):
        return state is trials.slam.tracking_state.TrackingState.LOST or (
            self._initializing_is_lost and state is trials.slam.tracking_state.TrackingState.NOT_INITIALIZED)

    def get_trial_requirements(self):
        return {'success': True, 'tracking_stats': {'$exists': True, '$ne': []}}

    def benchmark_results(self, trial_result):
        """
        Benchmark a trajectory 
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        ground_truth_traj = trial_result.get_ground_truth_camera_poses()
        states = trial_result.get_tracking_states()

        timestamps = list(ground_truth_traj.keys())
        timestamps.sort()

        currently_lost = False
        lost_start = 0
        lost_distance = 0
        prev_location = None
        total_distance = 0
        num_frames = 0
        lost_intervals = []

        for timestamp in timestamps:
            pose = ground_truth_traj[timestamp]
            distance = 0
            if prev_location is not None:
                distance = pose.location - prev_location
                distance = np.sqrt(np.dot(distance, distance))
                prev_location = pose.location
                total_distance += distance

            if self.is_lost(states[timestamp]):
                if currently_lost:
                    if distance is not None:
                        lost_distance += distance
                    prev_location = pose.location
                    num_frames += 1
                else:
                    currently_lost = True
                    lost_start = timestamp
                    lost_distance = 0
                    num_frames = 1
                    prev_location = pose.location
            elif currently_lost:
                currently_lost = False
                lost_distance += distance
                lost_intervals.append(LostInterval(start_time=lost_start,
                                                   end_time=timestamp,
                                                   distance=lost_distance,
                                                   num_frames=num_frames))
        # We're still lost at the end, add the final distance
        if currently_lost:
            lost_intervals.append(LostInterval(start_time=lost_start,
                                               end_time=timestamps[-1],
                                               distance=lost_distance,
                                               num_frames=num_frames))

        return benchmarks.tracking.tracking_result.TrackingBenchmarkResult(benchmark_id=self.identifier,
                                                                           trial_result_id=trial_result.identifier,
                                                                           lost_intervals=lost_intervals,
                                                                           total_distance=total_distance,
                                                                           total_time=timestamps[-1],
                                                                           total_frames=len(timestamps),
                                                                           settings=self.get_settings())
