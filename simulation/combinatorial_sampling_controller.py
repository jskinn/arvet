import collections
import core.sequence_type
import util.transform as tf
import dataset.import_image_source
import simulation.controller
import simulation.unrealcv.unrealcv_simulator as uecv_sim


class CombinatorialSampleController(simulation.controller.Controller):

    def __init__(self, x_samples, y_samples, z_samples, roll_samples, pitch_samples, yaw_samples,):
        self._x_samples = tuple(x_samples) if isinstance(x_samples, collections.Iterable) else (x_samples,)
        self._y_samples = tuple(y_samples) if isinstance(y_samples, collections.Iterable) else (y_samples,)
        self._z_samples = tuple(z_samples) if isinstance(z_samples, collections.Iterable) else (z_samples,)
        self._roll_samples = tuple(roll_samples) if isinstance(roll_samples, collections.Iterable) else (roll_samples,)
        self._pitch_samples = (tuple(pitch_samples) if isinstance(pitch_samples, collections.Iterable)
                               else (pitch_samples,))
        self._yaw_samples = tuple(yaw_samples) if isinstance(yaw_samples, collections.Iterable) else (yaw_samples,)

        self._x_index = 0
        self._y_index = 0
        self._z_index = 0
        self._roll_index = 0
        self._pitch_index = 0
        self._yaw_index = 0

    @property
    def motion_type(self):
        """
        Get the kind of image sequence produced by this controller.
        :return:
        """
        return core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

    def is_complete(self):
        """
        Is the motion for this controller complete.
        Some controllers will produce finite motion, some will not.
        Those that do not simply always return false here.
        :return:
        """
        return self._x_index >= len(self._x_samples)

    def get_next_pose(self):
        """
        Get the next camera pose to use.
        This needs to be a 6-dof transform, look at using a library like transformation3d
        :return:
        """
        if self.is_complete():
            return None
        pose = tf.Transform(location=(self._x_samples[self._x_index],
                                      self._y_samples[self._y_index],
                                      self._z_samples[self._z_index]),
                            rotation=(self._roll_samples[self._roll_index],
                                      self._pitch_samples[self._pitch_index],
                                      self._yaw_samples[self._yaw_index]))
        self._roll_index += 1
        if self._roll_index >= len(self._roll_samples):
            self._roll_index = 0
            self._pitch_index += 1
        if self._pitch_index >= len(self._pitch_samples):
            self._pitch_index = 0
            self._yaw_index += 1
        if self._yaw_index >= len(self._yaw_samples):
            self._yaw_index = 0
            self._z_index += 1
        if self._z_index >= len(self._z_samples):
            self._z_index = 0
            self._y_index += 1
        if self._y_index >= len(self._y_samples):
            self._y_index = 0
            self._x_index += 1
        return pose

    def reset(self):
        """
        Reset the controller state so we can restart the simulation
        :return: void
        """
        self._x_index = 0
        self._y_index = 0
        self._z_index = 0
        self._roll_index = 0
        self._pitch_index = 0
        self._yaw_index = 0


def generate_dataset(db_client, sim_config, x_samples, y_samples, z_samples, roll_samples, pitch_samples, yaw_samples,
                     fov_samples=90, aperture_samples=120):
    """
    Build a number of datasets sampling a series of parameters
    :param db_client: The database client
    :param sim_config: Configuration f
    :param x_samples:
    :param y_samples:
    :param z_samples:
    :param roll_samples:
    :param pitch_samples:
    :param yaw_samples:
    :param fov_samples:
    :param aperture_samples:
    :return:
    """
    controller = CombinatorialSampleController(x_samples, y_samples, z_samples,
                                               roll_samples, pitch_samples, yaw_samples)
    fov_samples = tuple(fov_samples) if isinstance(fov_samples, collections.Iterable) else (fov_samples,)
    aperture_samples = (tuple(aperture_samples) if isinstance(aperture_samples, collections.Iterable)
                        else (aperture_samples,))
    simulator = uecv_sim.UnrealCVSimulator(controller=controller, config=sim_config)
    simulator.set_autofocus(True)
    for fov in fov_samples:
        simulator.set_field_of_view(fov)
        for aperture in aperture_samples:
            simulator.set_fstop(aperture)
            controller.reset()
            dataset.import_image_source.import_dataset_from_image_source(db_client, simulator)
