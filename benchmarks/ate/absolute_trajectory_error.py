#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM, 2016 John Skinner, ACRV
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

import numpy as np
import util.associate as ass
import core.benchmark
import benchmarks.ate.ate_result


class BenchmarkATE(core.benchmark.Benchmark):
    """
    A tool for benchmarking classes using Absolute Trajectory Error (ATE)
    This is adapted from the evaluate_ate.py, distributed as part of the TUM RGBD benchmark tools.
    See: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
    """

    def __init__(self, offset=0, max_difference=0.02, scale=1.0, id_=None):
        """
        Create a Absolute Trajectory Error benchmark.

        There are 3 configuration properties for calculating ATE, which can be set as parameters:
        - offset: A uniform offset to the timstamps of the calculated trajectory, relative to the ground truth
        - max_difference: The maximum difference between matched timestamps
        - scale: A scaling factor between the test trajectory and the ground truth trajectory locations
        :param offset:
        :param max_difference:
        :param scale:
        """
        super().__init__(id_=id_)
        self._offset = offset
        self._max_difference = max_difference
        self._scale = scale     # TODO:

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
    def scale(self, scale):
        self._scale = scale

    @property
    def max_difference(self):
        return self._max_difference

    @max_difference.setter
    def max_difference(self, max_difference):
        if max_difference >= 0:
            self._max_difference = max_difference

    def get_settings(self):
        return {
            'offset': self.offset,
            'scale': self.scale,
            'max_difference': self.max_difference
        }

    def serialize(self):
        output = super().serialize()
        output['offset'] = self.offset
        output['max_difference'] = self.max_difference
        output['scale'] = self.scale
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'offset' in serialized_representation:
            kwargs['offset'] = serialized_representation['offset']
        if 'max_difference' in serialized_representation:
            kwargs['max_difference'] = serialized_representation['max_difference']
        if 'scale' in serialized_representation:
            kwargs['scale'] = serialized_representation['scale']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    def get_trial_requirements(self):
        return {'success': True, 'trajectory': {'$exists': True, '$ne': []}}

    def benchmark_results(self, trial_result):
        """

        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        ground_truth_traj = trial_result.get_ground_truth_camera_poses()
        result_traj = trial_result.get_computed_camera_poses()
        # TODO: Fix the format of these trajectories to what is expected below
        matches = ass.associate(ground_truth_traj, result_traj, self.offset, self.max_difference)
        if len(matches) < 2:
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  "Couldn't find matching timestamp pairs "
                                                  "between groundtruth and estimated trajectory! "
                                                  "Did you choose the correct sequence?")

        # Construct matrices of the ground truth and calculated poses
        ground_truth_xyz = []
        result_xyz = []
        gt_timestamps = []
        for gt_stamp, result_stamp in matches:
            ground_truth_xyz.append(ground_truth_traj[gt_stamp].location)
            result_xyz.append(result_traj[result_stamp].location * float(self.scale))
            gt_timestamps.append(gt_stamp)

        # Align the two trajectories, based on the matching timestamps from both
        ground_truth_xyz = np.matrix(ground_truth_xyz).transpose()
        result_xyz = np.matrix(result_xyz).transpose()
        rot, trans, trans_error = align(result_xyz, ground_truth_xyz)

        # Match the trans error back to its ground-truth timestamps
        mapped_error = {}
        for col_num, timestamp in enumerate(gt_timestamps):
            mapped_error[timestamp] = trans_error[col_num]

        return benchmarks.ate.ate_result.BenchmarkATEResult(self.identifier, trial_result.identifier,
                                                            mapped_error, self.get_settings())


def trajectory_to_dict(trajectory):
    return {point.timestamp: [
        point.location[0], point.location[1], point.location[2],
        point.orientation[0], point.orientation[1], point.orientation[2], point.orientation[3]
    ] for point in trajectory}


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error
