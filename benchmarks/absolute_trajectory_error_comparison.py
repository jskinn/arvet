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
#
# Requirements:
# sudo apt-get install python-argparse

import numpy
import rgbd_benchmark_tools.associate as associate
import core.benchmark
import core.comparison
from core.visual_slam import SLAMTrialResult


class BenchmarkATEComparisonResult(core.comparison.ComparisonBenchmarkResult):
    """
    Average Trajectory Error results.

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id, trial_result_id, reference_id, translational_error, ate_settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, reference_id, True, id_)

        self._trans_error = translational_error

        if isinstance(translational_error, dict):
            self._num_pairs = translational_error['num_pairs']
            self._rmse = translational_error['rmse']
            self._mean = translational_error['mean']
            self._median = translational_error['median']
            self._std = translational_error['std']
            self._min = translational_error['min']
            self._max = translational_error['max']
        else:
            self._num_pairs = len(translational_error)
            self._rmse = numpy.sqrt(numpy.dot(translational_error, translational_error) / self._num_pairs)
            self._mean = numpy.mean(translational_error)
            self._median = numpy.median(translational_error)
            self._std = numpy.std(translational_error)
            self._min = numpy.min(translational_error)
            self._max = numpy.max(translational_error)

        self._ate_settings = ate_settings

    @property
    def num_pairs(self):
        return len(self._trans_error)

    @property
    def rmse(self):
        return numpy.sqrt(numpy.dot(self._trans_error, self._trans_error) / self.num_pairs)

    @property
    def mean(self):
        return numpy.mean(self._trans_error)

    @property
    def median(self):
        return numpy.median(self._trans_error)

    @property
    def std(self):
        return numpy.std(self._trans_error)

    @property
    def min(self):
        return numpy.min(self._trans_error)

    @property
    def max(self):
        return numpy.max(self._trans_error)

    @property
    def settings(self):
        return self._ate_settings

    def serialize(self):
        output = super().serialize()
        output['trans_error'] = self._trans_error.tolist()
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'trans_error' in serialized_representation:
            kwargs['translational_error'] = numpy.array(serialized_representation['trans_error'])
        if 'settings' in serialized_representation:
            kwargs['ate_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkATEComparison(core.comparison.ComparisonBenchmark):
    """
    A tool for benchmarking classes using Absolute Trajectory Error (ATE)
    """

    def __init__(self, offset=0, max_difference=0.02, scale=1.0):
        """
        Create a Average Trajectory Error benchmark.

        There are 3 configuration properties for ATE calculating ATE, which can be set as parameters:
        - offset: A uniform offset to the timstamps of the calculated trajectory, relative to the
        - max_difference: The maximum difference betwee
        :param offset:
        :param max_difference:
        :param scale:
        """
        self._offset = offset
        self._max_difference = max_difference
        self._scale = scale

    @property
    def identifier(self):
        return 'AbsoluteTrajectoryErrorComparison'

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

    def get_trial_requirements(self):
        return {'success': True, 'trajectory': { '$exists': True, '$ne': []}}

    def compare_results(self, trial_result, reference_trial_result, reference_dataset_images):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if not isinstance(trial_result, SLAMTrialResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier, 'Trial was not a slam trial')
        if not isinstance(reference_trial_result, SLAMTrialResult):
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier, 'Reference trial was not a slam trial')

        ground_truth_dict = trajectory_to_dict(reference_dataset_images.get_ground_truth_trajectory(trial_result.dataset_repeats))
        reference_dict = trajectory_to_dict(reference_trial_result.trajectory)
        trial_dict = trajectory_to_dict(trial_result.trajectory)

        comparison_matches = associate.associate(reference_dict, trial_dict, self.offset, self.max_difference)
        if len(comparison_matches) < 2:
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  "Couldn't find matching timestamp pairs "
                                                  "between reference and trial trajectory! ")
        reference_dict = {time : reference_dict[time] for time, _ in comparison_matches}
        trial_dict = {time: trial_dict[time] for _, time in comparison_matches}

        gt_matches = associate.associate(ground_truth_dict, reference_dict, self.offset, self.max_difference)
        if len(gt_matches) < 2:
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  "Couldn't find matching timestamp pairs "
                                                  "between groundtruth and reference trajectory! ")

        # Construct matrices of the ground trugth and calculated
        ground_truth_xyz = numpy.matrix([[float(value)
                                          for value
                                          in ground_truth_dict[a][0:3]]
                                         for a, b
                                         in gt_matches]).transpose()
        ref_xyz = numpy.matrix([[float(value) * float(self.scale)
                                    for value
                                    in reference_dict[b][0:3]]
                                   for a, b
                                   in gt_matches]).transpose()
        trial_xyz = numpy.matrix([[float(value) * float(self.scale)
                                 for value
                                 in trial_dict[b][0:3]]
                                for a, b
                                in gt_matches]).transpose()

        # Align the two trajectories, based on the matching timestamps from both
        _, _, reference_trans_error = align(ref_xyz, ground_truth_xyz)
        _, _, trial_trans_error = align(trial_xyz, ground_truth_xyz)
        error_change = trial_trans_error - reference_trans_error

        return BenchmarkATEComparisonResult(self.identifier, trial_result.identifier, reference_trial_result.identifier, error_change, self.get_settings())


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
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if (numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error
