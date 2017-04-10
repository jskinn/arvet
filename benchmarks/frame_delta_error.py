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
This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.
"""

import copy
import numpy
import core.benchmark
import libviso2.libviso2


# TODO: This is the same as the relative pose error, merge and make it work for both trajectories and frame deltas.
class BenchmarkFrameDeltaResult(core.benchmark.BenchmarkResult):
    """
    Relative Pose Error results.

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id, trial_result_id, translational_error,
                 rotational_error, rpe_settings, id_=None, **kwargs):
        super().__init__(benchmark_id, trial_result_id, True, id_)

        self._translational_error = translational_error
        self._rotational_error = rotational_error
        self._rpe_settings = rpe_settings

    @property
    def num_pairs(self):
        return len(self._translational_error)

    @property
    def translational_error(self):
        return self._translational_error

    @property
    def trans_rmse(self):
        return numpy.sqrt(numpy.dot(self.translational_error, self.translational_error) / self.num_pairs)

    @property
    def trans_mean(self):
        return numpy.mean(self.translational_error)

    @property
    def trans_median(self):
        return numpy.median(self.translational_error)

    @property
    def trans_std(self):
        return numpy.std(self.translational_error)

    @property
    def trans_min(self):
        return numpy.min(self.translational_error)

    @property
    def trans_max(self):
        return numpy.max(self.translational_error)

    @property
    def rotational_error(self):
        return self._rotational_error

    @property
    def rot_rmse(self):
        return numpy.sqrt(numpy.dot(self.rotational_error, self.rotational_error) / self.num_pairs)

    @property
    def rot_mean(self):
        return numpy.mean(self.rotational_error)

    @property
    def rot_median(self):
        return numpy.median(self.rotational_error)

    @property
    def rot_std(self):
        return numpy.std(self.rotational_error)

    @property
    def rot_min(self):
        return numpy.min(self.rotational_error)

    @property
    def rot_max(self):
        return numpy.max(self.rotational_error)

    @property
    def settings(self):
        return self._rpe_settings

    def serialize(self):
        output = super().serialize()
        output['translational_error'] = self.translational_error.tolist()
        output['rotational_error'] = self.rotational_error.tolist()
        output['settings'] = copy.deepcopy(self.settings)
        return output

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
        if 'translational_error' in serialized_representation:
            kwargs['translational_error'] = numpy.array(serialized_representation['translational_error'])
        if 'rotational_error' in serialized_representation:
            kwargs['rotational_error'] = numpy.array(serialized_representation['rotational_error'])
        if 'settings' in serialized_representation:
            kwargs['rpe_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, **kwargs)


class BenchmarkFrameDelta(core.benchmark.Benchmark):

    def __init__(self):
        pass

    @property
    def identifier(self):
        return 'RelativePoseErrorFrameDelta'

    def get_settings(self):
        return {}

    def get_trial_requirements(self):
        return {'success': True, 'frame_deltas': {'$exists': True, '$ne': []}}

    def benchmark_results(self, dataset_images, trial_result):
        """

        :param dataset_images: The dataset of images
        :param trial_result: The results of a particular trial
        :return:
        :rtype BenchmarkResult:
        """
        if isinstance(trial_result, libviso2.libviso2.LibVisOTrialResult):
            trans_error = []
            rot_error = []
            for idx, s_frame_delta in enumerate(trial_result.frame_deltas):
                if idx == 0:
                    continue
                frame_delta = numpy.array(s_frame_delta)
                current_frame_transform = make_transform(dataset_images[idx].camera_location,
                                                         dataset_images[idx].camera_orientation)
                prev_frame_transform = make_transform(dataset_images[idx - 1].camera_location,
                                                      dataset_images[idx - 1].camera_orientation)

                error44 = ominus(frame_delta, ominus(current_frame_transform, prev_frame_transform))
                trans = compute_distance(error44)
                rot = compute_angle(error44)

                trans_error.append(trans)
                rot_error.append(rot)

            return BenchmarkFrameDeltaResult(benchmark_id=self.identifier,
                                             trial_result_id=trial_result.identifier,
                                             translational_error=numpy.array(trans_error),
                                             rotational_error=numpy.array(rot_error),
                                             rpe_settings=self.get_settings())
        else:
            return core.benchmark.FailedBenchmark(self.identifier, trial_result.identifier,
                                                  'Trial was not a libviso2 trial')


_EPS = numpy.finfo(float).eps * 4.0


def make_transform(translation, orientation):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    translation -- numpy vector containing the translation.
    orientation -- unit quaternion describing the orientation

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    q = orientation
    nq = numpy.dot(orientation, orientation)
    if nq < _EPS:
        return numpy.array((
            (1.0, 0.0, 0.0, translation[0]),
            (0.0, 1.0, 0.0, translation[1]),
            (0.0, 0.0, 1.0, translation[2]),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], translation[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], translation[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], translation[2]),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=numpy.float64)


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
    sum_ = 0
    for t in motion:
        sum_ += compute_distance(t)
        distances.append(sum_)
    return distances


def rotations_along_trajectory(traj, angle_scale):
    """
    Compute the angular rotations along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    sum_ = 0
    for t in motion:
        sum_ += compute_angle(t) * angle_scale
        distances.append(sum_)
    return distances
