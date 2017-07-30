import numpy as np
import core.benchmark
import util.associate
import benchmarks.trajectory_drift.trajectory_drift_result as drif_result


class BenchmarkTrajectoryDrift(core.benchmark.Benchmark):
    """
    Measure VO/SLAM performance by the drift in segments of different lengths.
    This is based on a python port of the KITTI odometry evaluation metric used on:
    http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    which is implemented in "evaluation_odometry.cpp", distributed in "odometry development kit",
    on the same page.
    """

    def __init__(self, segment_lengths=None, step_size=10, id_=None):
        """


        :param segment_lengths: A list of different trajectory segment lengths to evaluate.
        Default [100, 200, 300, 400, 500, 600, 700, 800]
        """
        super().__init__(id_=id_)
        self._segment_lengths = (list(segment_lengths) if segment_lengths is not None
                                 else [100, 200, 300, 400, 500, 600, 700, 800])
        self._step_size = int(step_size)

    def get_settings(self):
        return {
            'segment_lengths': self._segment_lengths,
            'step_size': self._step_size
        }

    def serialize(self):
        output = super().serialize()
        output['segment_lengths'] = self._segment_lengths
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'segment_lengths' in serialized_representation:
            kwargs['segment_lengths'] = serialized_representation['segment_lengths']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_trial_requirements(cls):
        return {'success': True}

    def is_trial_appropriate(self, trial_result):
        return (hasattr(trial_result, 'identifier') and
                hasattr(trial_result, 'get_ground_truth_camera_poses') and
                hasattr(trial_result, 'computed_camera_poses'))

    def benchmark_results(self, trial_result):
        """

        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        ground_truth_traj = trial_result.get_ground_truth_camera_poses()
        result_traj = trial_result.get_computed_camera_poses()

        ground_truth_traj = {stamp: pose.transform_matrix for stamp, pose in ground_truth_traj.items()}
        result_traj = {stamp: pose.transform_matrix for stamp, pose in result_traj.items()}

        # TODO: Configure association?
        matches = util.associate.associate(ground_truth_traj, result_traj, offset=0, max_difference=1)
        if matches is None or len(matches) < 2:
            return core.benchmark.FailedBenchmark(benchmark_id=self.identifier,
                                                  trial_result_id=trial_result.identifier,
                                                  reason="Couldn't find matching timestamp pairs between"
                                                         "groundtruth and estimated trajectory!")

        gt_poses = [ground_truth_traj[match[0]].transform_matrix for match in matches]
        result_poses = [result_traj[match[1]].transform_matrix for match in matches]
        errors = calc_sequence_errors(gt_poses, result_poses, segment_lengths=self._segment_lengths, step_size=10)

        return drif_result.TrajectoryDriftBenchmarkResult(benchmark_id=self.identifier,
                                                          trial_result_id=trial_result.identifier,
                                                          errors=errors, settings=self.get_settings())


def calc_sequence_errors(poses_gt, poses_result, segment_lengths, step_size=10):
    """
    Find the error in a list of computed poses, over different segment lengths
    Based on "calcSequenceErrors" in "evaluate_odometry.cpp" ln 81
    :param poses_gt: The list of ground truth poses, in order. Must be a list and not frames
    :param poses_result: The list of computed poses, in order. Must be a list of the same length as poses_gt
    :param segment_lengths: The list of segment lengths to test
    :param step_size: The step size between start frames when choosing segments. Default 10.
    :return: A list of dictionaries containing the computed rotational and translational errors
    """
    err = []    # error vector
    dist = trajectory_distances(poses_gt)    # pre - compute distances from ground truth as reference)

    # for all start positions do
    for first_frame in range(0, len(poses_gt), step_size):
        # for all segment lengths do
        for i in range(0, len(segment_lengths)):
            # current length
            seg_len = segment_lengths[i]

            # compute last frame
            last_frame = last_frame_from_segment_length(dist, first_frame, seg_len)

            # continue, if sequence not long enough
            if last_frame <= -1:
                    continue

            # compute rotational and translational errors
            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) * poses_gt[last_frame]
            pose_delta_result = np.linalg.inv(poses_result[first_frame]) * poses_result[last_frame]
            pose_error = np.linalg.inv(pose_delta_result) * pose_delta_gt
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)

            # compute speed
            num_frames = last_frame - first_frame + 1
            speed = seg_len / (0.1 * num_frames)

            # write to file
            err.append({
                'first_frame': first_frame,
                'r_err': r_err / seg_len,
                't_err': t_err / seg_len,
                'len': seg_len,
                'speed': speed
            })

    # return error vector
    return err


def trajectory_distances(poses):
    """
    Compute the distance along the trajectory for each recorded pose.
    Based on "trajectoryDistances" in "evaluate_odometry.cpp" ln 45
    :param poses: A list of poses in the order they occurred
    :return: A list of the total trajectory length for each pose in the provided length
    """
    dist = [0]
    for i in range(1, len(poses)):
        p1 = poses[i-1]
        p2 = poses[i]
        dx = p1.val[0][3]-p2.val[0][3]
        dy = p1.val[1][3]-p2.val[1][3]
        dz = p1.val[2][3]-p2.val[2][3]
        dist.append(dist[i-1] + np.sqrt(dx * dx + dy * dy + dz * dz))
    return dist


def last_frame_from_segment_length(dist, first_frame, seg_len):
    """
    Given a desired segment length, find the last frame such that the distance between
    between the first and last frame is just greater than the segment length.
    Based on "lastFrameFromSegmentLength" in "evaluate_odometry.cpp" ln 59
    :param dist: A list of distances through the trajectory at each frame. See "trajectory_distances", above.
    :param first_frame: The index of the first frame
    :param seg_len:
    :return: The index of the last frame, or -1 if there is not enough distance left in teh trajectory.
    """
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + seg_len:
            return i
    return -1


def rotation_error(pose_error):
    """
    Compute the rotation error for a given error matrix.
    Based on "rotationError" in "evaluate_odometry.cpp" ln 66
    :param pose_error:
    :return: floating point rotation error for this pose error matrix
    """
    a = pose_error[0][0]
    b = pose_error[1][1]
    c = pose_error[2][2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(pose_error):
    """
    Compute the translation error for a given error matrix
    Based on "translationError" in "evaluate_odometry.cpp" ln 74
    :param pose_error:
    :return:
    """
    dx = pose_error[0][3]
    dy = pose_error[1][3]
    dz = pose_error[2][3]
    return np.sqrt(dx * dx + dy * dy + dz * dz)
