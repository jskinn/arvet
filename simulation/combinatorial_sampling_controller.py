import collections
import itertools
import numpy as np
import core.sequence_type
import util.transform as tf
import simulation.controller


class CombinatorialSampleController(simulation.controller.Controller):

    def __init__(self, x_samples, y_samples, z_samples, roll_samples, pitch_samples, yaw_samples,
                 fov_samples=90, aperture_samples=120, subject_pose=None, proximity_distance=50):
        self._x_samples = tuple(x_samples) if isinstance(x_samples, collections.Iterable) else (x_samples,)
        self._y_samples = tuple(y_samples) if isinstance(y_samples, collections.Iterable) else (y_samples,)
        self._z_samples = tuple(z_samples) if isinstance(z_samples, collections.Iterable) else (z_samples,)
        self._roll_samples = tuple(roll_samples) if isinstance(roll_samples, collections.Iterable) else (roll_samples,)
        self._pitch_samples = (tuple(pitch_samples) if isinstance(pitch_samples, collections.Iterable)
                               else (pitch_samples,))
        self._yaw_samples = tuple(yaw_samples) if isinstance(yaw_samples, collections.Iterable) else (yaw_samples,)
        self._fov_samples = tuple(fov_samples) if isinstance(fov_samples, collections.Iterable) else (fov_samples,)
        self._aperture_samples = (tuple(aperture_samples) if isinstance(aperture_samples, collections.Iterable)
                                  else (aperture_samples,))

        self._subject_pose = subject_pose
        self._proximity_threshold = proximity_distance
        self._sample_filter = None

        self._change_fov = True
        self._change_aperture = True
        self._next_settings = None
        self._settings_iterator = itertools.product(self._fov_samples, self._aperture_samples, self._x_samples,
                                                    self._y_samples, self._z_samples, self._roll_samples,
                                                    self._pitch_samples, self._yaw_samples)
        self.find_next_camera_settings()

    @property
    def motion_type(self):
        """
        Get the kind of image sequence produced by this controller.
        :return: ImageSequenceType.NON_SEQUENTIAL
        """
        return core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

    def is_complete(self):
        """
        Is the motion for this controller complete.
        Some controllers will produce finite motion, some will not.
        Those that do not simply always return false here.
        :return:
        """
        return self._next_settings is None or self._settings_iterator is None

    def update_state(self, delta_time, simulator):
        """
        Update the simulator to the next sample point.
        :return:
        """
        if not self.is_complete():
            fov, aperture, x, y, z, roll, pitch, yaw = self._next_settings
            if self._change_fov:
                simulator.set_field_of_view(fov)
            if self._change_aperture:
                simulator.set_fstop(aperture)
            simulator.set_camera_pose(tf.Transform(location=(x, y, z), rotation=(roll, pitch, yaw)))
            self.find_next_camera_settings()

    def reset(self):
        """
        Reset the controller state so we can restart the simulation
        :return: void
        """
        self._settings_iterator = itertools.product(self._fov_samples, self._aperture_samples, self._x_samples,
                                                    self._y_samples, self._z_samples, self._roll_samples,
                                                    self._pitch_samples, self._yaw_samples)
        self.find_next_camera_settings()

    def find_next_camera_settings(self):
        """
        Find the next sample pose in the iteration.
        We have to do this so that we know if we're complete,
        if we wait for update_state to get the next iteration state,
        we don't know if there will be one.
        :return:
        """
        candidate_settings = None
        can_see_subject = False
        while self._settings_iterator is not None and not can_see_subject:
            try:
                candidate_settings = next(self._settings_iterator)
            except StopIteration:
                self._settings_iterator = None
                candidate_settings = None
                break
            fov, aperture, x, y, z, roll, pitch, yaw = candidate_settings
            new_camera_pose = tf.Transform(location=(x, y, z), rotation=(roll, pitch, yaw))
            can_see_subject = self.can_see_from_pose(new_camera_pose, fov)
        self._change_fov = (candidate_settings is not None and
                            (self._next_settings is None or self._next_settings[0] != candidate_settings[0]))
        self._change_aperture = (candidate_settings is not None and
                                 (self._next_settings is None or self._next_settings[1] != candidate_settings[1]))
        self._next_settings = candidate_settings


    def can_see_from_pose(self, pose, fov):
        """
        Can the subject of this controller be seen from the given location.
        The sampling controller will only use poses that pass this test
        :param pose:
        :param fov:
        :return:
        """
        if self._subject_pose is None:
            # We don't have a pose for the subject, this check is useless.
            return True
        relative_subject_pose = pose.find_relative(self._subject_pose)
        if relative_subject_pose.location[0] < 0:
            # Object is behind the camera, we can't see it.
            return False

        distance = np.linalg.norm(relative_subject_pose.location)
        if distance < self._proximity_threshold:
            # We are too close to the object to see it, we might be inside it
            return False

        # Take the dot product with (1,0,0), and divide by the product of their magnitudes.
        # This reduces to location.x divided by the norm of the location,
        # |a||b|*cos_theta = dot(a, b)
        # cos_theta = dot(a, b) / |a||b|
        # cos_theta = dot(loc, (1,0,0)) / (|loc| * 1)
        # cos_theta = loc.x / |loc|
        angle = abs(np.arccos(relative_subject_pose.location[0] / distance))
        angle = angle * 180 / np.pi     # Convert to degrees, because the field of view is in degrees
        return angle < fov/2
