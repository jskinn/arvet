#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
This is based on the code in evaluate_rpe.py distributed in the TUM RGBD benchmark tools.
See: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
Original Comment:
This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.
"""

import random
import numpy
import core.benchmark
import benchmarks.rpe.rpe_result


class BenchmarkRPE(core.benchmark.Benchmark):

    def __init__(self, max_pairs=10000, fixed_delta=False, delta=1.0, delta_unit='s', offset=0, scale_=1, id_=None):
        """

        :param max_pairs: maximum number of pose comparisons (default: 10000, set to zero to disable downsampling
        :param fixed_delta: only consider pose pairs that have a distance of delta delta_unit
        (e.g., for evaluating the drift per second/meter/radian)
        :param delta: delta for evaluation (default: 1.0)
        :param delta_unit: unit of delta
        (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'s\')
        :param offset: time offset between ground-truth and estimated trajectory (default: 0.0)
        :param scale_: scaling factor for the estimated trajectory (default: 1.0)
        """
        super().__init__(id_=id_)
        self._max_pairs = int(max_pairs)
        self._fixed_delta = fixed_delta
        self._delta = delta
        if delta_unit is 's' or delta_unit is 'm' or delta_unit is 'rad' or delta_unit is 'f':
            self._delta_unit = delta_unit
        else:
            self._delta_unit = 's'
        self._offset = offset
        self._scale = scale_

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale_):
        self._scale = scale_

    @property
    def max_pairs(self):
        return self._max_pairs

    @max_pairs.setter
    def max_pairs(self, max_pairs):
        self._max_pairs = max_pairs

    @property
    def fixed_delta(self):
        return self._fixed_delta

    @fixed_delta.setter
    def fixed_delta(self, fixed_delta):
        self._fixed_delta = fixed_delta

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._delta = delta

    @property
    def delta_unit(self):
        return self._delta_unit

    @delta_unit.setter
    def delta_unit(self, delta_unit):
        if delta_unit is 's' or delta_unit is 'm' or delta_unit is 'rad' or delta_unit is 'f':
            self._delta_unit = delta_unit

    def get_settings(self):
        return {
            'offset': self.offset,
            'scale': self.scale,
            'max_pairs': self.max_pairs,
            'fixed_delta': self.fixed_delta,
            'delta': self.delta,
            'delta_unit': self.delta_unit
        }

    def serialize(self):
        output = super().serialize()
        output['offset'] = self.offset
        output['scale'] = self.scale
        output['max_pairs'] = self.max_pairs
        output['fixed_delta'] = self.fixed_delta
        output['delta'] = self.delta
        output['delta_unit'] = self.delta_unit
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'offset' in serialized_representation:
            kwargs['offset'] = serialized_representation['offset']
        if 'scale' in serialized_representation:
            kwargs['scale_'] = serialized_representation['scale']
        if 'max_pairs' in serialized_representation:
            kwargs['max_pairs'] = serialized_representation['max_pairs']
        if 'fixed_delta' in serialized_representation:
            kwargs['fixed_delta'] = serialized_representation['fixed_delta']
        if 'delta' in serialized_representation:
            kwargs['delta'] = serialized_representation['delta']
        if 'delta_unit' in serialized_representation:
            kwargs['delta_unit'] = serialized_representation['delta_unit']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    def get_trial_requirements(self):
        return {}

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

        result = evaluate_trajectory(traj_gt=ground_truth_traj,
                                     traj_est=result_traj,
                                     param_max_pairs=int(self.max_pairs),
                                     param_fixed_delta=self.fixed_delta,
                                     param_delta=float(self.delta),
                                     param_delta_unit=self.delta_unit,
                                     param_offset=float(self.offset),
                                     param_scale=self.scale)
        if result is None or len(result) < 2:
            return core.benchmark.FailedBenchmark(benchmark_id=self.identifier,
                                                  trial_result_id=trial_result.identifier,
                                                  reason="Couldn't find matching timestamp pairs between"
                                                         "groundtruth and estimated trajectory!")

        result = numpy.array(result)
        gt_post_timestamps = result[:, 3]
        trans_error = result[:, 4]
        rot_error = result[:, 5]
        trans_error = dict(zip(gt_post_timestamps, trans_error))
        rot_error = dict(zip(gt_post_timestamps, rot_error))
        return benchmarks.rpe.rpe_result.BenchmarkRPEResult(self.identifier, trial_result.identifier,
                                                            trans_error, rot_error, self.get_settings())


def find_closest_index(L, t):
    """
    Find the index of the closest value in a list.

    Input:
    L -- the list
    t -- value to be found

    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end + beginning) / 2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best


def ominus(a, b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return numpy.dot(numpy.linalg.inv(a), b)


def scale(a, scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return numpy.array(
        [[a[0, 0], a[0, 1], a[0, 2], a[0, 3] * scalar],
         [a[1, 0], a[1, 1], a[1, 2], a[1, 3] * scalar],
         [a[2, 0], a[2, 1], a[2, 2], a[2, 3] * scalar],
         [a[3, 0], a[3, 1], a[3, 2], a[3, 3]]]
    )


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return numpy.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return numpy.arccos(min(1, max(-1, (numpy.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    total = 0
    for t in motion:
        total += compute_distance(t)
        distances.append(total)
    return distances


def rotations_along_trajectory(traj, scale_):
    """
    Compute the angular rotations along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    total = 0
    for t in motion:
        total += compute_angle(t) * scale_
        distances.append(total)
    return distances


def evaluate_trajectory(traj_gt, traj_est, param_max_pairs=10000, param_fixed_delta=False, param_delta=1.00,
                        param_delta_unit="s", param_offset=0.00, param_scale=1.00):
    """
    Compute the relative pose error between two trajectories.

    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory

    Output:
    list of compared poses and the resulting translation and rotation error
    """
    stamps_gt = list(traj_gt.keys())
    stamps_est = list(traj_est.keys())
    stamps_gt.sort()
    stamps_est.sort()

    stamps_est_return = []
    for t_est in stamps_est:
        t_gt = stamps_gt[find_closest_index(stamps_gt, t_est + param_offset)]
        t_est_return = stamps_est[find_closest_index(stamps_est, t_gt - param_offset)]
        # t_gt_return = stamps_gt[find_closest_index(stamps_gt, t_est_return + param_offset)]
        if t_est_return not in stamps_est_return:
            stamps_est_return.append(t_est_return)
    if len(stamps_est_return) < 2:
        return None

    if param_delta_unit == "s":
        index_est = list(traj_est.keys())
        index_est.sort()
    elif param_delta_unit == "m":
        index_est = distances_along_trajectory(traj_est)
    elif param_delta_unit == "rad":
        index_est = rotations_along_trajectory(traj_est, 1)
    elif param_delta_unit == "deg":
        index_est = rotations_along_trajectory(traj_est, 180 / numpy.pi)
    elif param_delta_unit == "f":
        index_est = range(len(traj_est))
    else:
        raise Exception("Unknown unit for delta: '%s'" % param_delta_unit)

    if not param_fixed_delta:
        if param_max_pairs == 0 or len(traj_est) < numpy.sqrt(param_max_pairs):
            pairs = [(i, j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0, len(traj_est) - 1), random.randint(0, len(traj_est) - 1))
                     for _ in range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = find_closest_index(index_est, index_est[i] + param_delta)
            if j != len(traj_est) - 1:
                pairs.append((i, j))
        if param_max_pairs != 0 and len(pairs) > param_max_pairs:
            pairs = random.sample(pairs, param_max_pairs)

    gt_interval = numpy.median([s - t for s, t in zip(stamps_gt[1:], stamps_gt[:-1])])
    gt_max_time_difference = 2 * gt_interval

    result = []
    for i, j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]

        stamp_gt_0 = stamps_gt[find_closest_index(stamps_gt, stamp_est_0 + param_offset)]
        stamp_gt_1 = stamps_gt[find_closest_index(stamps_gt, stamp_est_1 + param_offset)]

        if (abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference or
                abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
            continue

        error44 = ominus(scale(
            ominus(traj_est[stamp_est_1], traj_est[stamp_est_0]), param_scale),
            ominus(traj_gt[stamp_gt_1], traj_gt[stamp_gt_0]))

        trans = compute_distance(error44)
        rot = compute_angle(error44)

        result.append([stamp_est_0, stamp_est_1, stamp_gt_0, stamp_gt_1, trans, rot])
    return result
