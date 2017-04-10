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
from core.benchmark import Benchmark, BenchmarkResult, FailedBenchmark
from core.visual_slam import SLAMTrialResult


class BenchmarkATEResult(BenchmarkResult):
    """
    Average Trajectory Error results.

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id, trial_result_id, translational_error, ate_settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, True, id_)

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
        return self._num_pairs

    @property
    def rmse(self):
        return self._rmse

    @property
    def mean(self):
        return self._mean

    @property
    def median(self):
        return self._median

    @property
    def std(self):
        return self._std

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def settings(self):
        return self._ate_settings

    def serialize(self):
        output = super().serialize()
        output['num_pairs'] = self.num_pairs
        output['rmse'] = self.rmse
        output['mean'] = self.mean
        output['median'] = self.median
        output['std'] = self.std
        output['min'] = self.min
        output['max'] = self.max
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        translational_error = {}
        if 'num_pairs' in serialized_representation:
            translational_error['num_pairs'] = serialized_representation['num_pairs']
        if 'rmse' in serialized_representation:
            translational_error['rmse'] = serialized_representation['rmse']
        if 'mean' in serialized_representation:
            translational_error['mean'] = serialized_representation['mean']
        if 'median' in serialized_representation:
            translational_error['median'] = serialized_representation['median']
        if 'std' in serialized_representation:
            translational_error['std'] = serialized_representation['std']
        if 'min' in serialized_representation:
            translational_error['min'] = serialized_representation['min']
        if 'max' in serialized_representation:
            translational_error['max'] = serialized_representation['max']
        kwargs['translational_error'] = translational_error

        if 'settings' in serialized_representation:
            kwargs['ate_settings'] = serialized_representation['settings']

        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkATE(Benchmark):
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
        return 'AverageTrajectoryError'  # I got this name wrong.

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

    def benchmark_results(self, dataset_images, trial_result):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if isinstance(trial_result, SLAMTrialResult):
            ground_truth_dict = trajectory_to_dict(dataset_images.get_ground_truth_trajectory(trial_result.dataset_repeats))
            result_dict = trajectory_to_dict(trial_result.trajectory)

            matches = associate.associate(ground_truth_dict, result_dict, self.offset, self.max_difference)
            if len(matches) < 2:
                return FailedBenchmark(self.identifier, trial_result.identifier,
                                       "Couldn't find matching timestamp pairs "
                                       "between groundtruth and estimated trajectory! "
                                       "Did you choose the correct sequence?")

            # Construct matrices of the ground trugth and calculated
            ground_truth_xyz = numpy.matrix([[float(value)
                                              for value
                                              in ground_truth_dict[a][0:3]]
                                             for a, b
                                             in matches]).transpose()
            result_xyz = numpy.matrix([[float(value) * float(self.scale)
                                        for value
                                        in result_dict[b][0:3]]
                                       for a, b
                                       in matches]).transpose()

            # Align the two trajectories, based on the matching timestamps from both
            rot, trans, trans_error = align(result_xyz, ground_truth_xyz)

            return BenchmarkATEResult(self.identifier, trial_result.identifier, trans_error, self.get_settings())
        else:
            return FailedBenchmark(self.identifier, trial_result.identifier, 'Trial was not a slam trial')


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
