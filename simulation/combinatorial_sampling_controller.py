import collections
import itertools
import core.sequence_type
import util.transform as tf
import simulation.controller


class CombinatorialSampleController(simulation.controller.Controller):

    def __init__(self, x_samples, y_samples, z_samples, roll_samples, pitch_samples, yaw_samples,
                 fov_samples=90, aperture_samples=120):
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

        self._settings_iterator = itertools.product(self._fov_samples, self._aperture_samples, self._x_samples,
                                                    self._y_samples, self._z_samples, self._roll_samples,
                                                    self._pitch_samples, self._yaw_samples)
        self._next_settings = next(self._settings_iterator)
        self._change_fov = True
        self._change_aperture = True

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
        return self._next_settings is not None

    def update_state(self, delta_time, simulator):
        """
        Update the simulator to the next sample point.
        :return:
        """
        if self.is_complete():
            return

        fov, aperture, x, y, z, roll, pitch, yaw = self._next_settings

        if self._change_fov:
            simulator.set_field_of_view(fov)
        if self._change_aperture:
            simulator.set_fstop(aperture)
        simulator.set_camera_pose(tf.Transform(location=(x, y, z),
                                               rotation=(roll, pitch, yaw)))
        try:
            self._next_settings = next(self._settings_iterator)
        except StopIteration:
            self._settings_iterator = None
            self._next_settings = None
        self._change_fov = bool(self._next_settings is not None and fov != self._next_settings[0])
        self._change_aperture = bool(self._next_settings is not None and aperture != self._next_settings[1])

    def reset(self):
        """
        Reset the controller state so we can restart the simulation
        :return: void
        """
        self._settings_iterator = itertools.product(self._fov_samples, self._aperture_samples, self._x_samples,
                                                    self._y_samples, self._z_samples, self._roll_samples,
                                                    self._pitch_samples, self._yaw_samples)
        self._next_settings = next(self._settings_iterator)
