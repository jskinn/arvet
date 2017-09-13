#Copyright (c) 2017, John Skinner
import collections
import itertools
import numpy as np
import core.sequence_type
import util.transform as tf
import simulation.simulator
import simulation.controller


class CombinatorialSampleController(simulation.controller.Controller):
    """
    A controller that loops through every combination of samples over several properties.
    This takes lists of values for each property, and iterates through the
    cartesian product of these values.
    If a subject pose is provided, it will filter the sample points to only those that can see the
    Naturally, this is non-sequential.
    """

    def __init__(self, x_samples, y_samples, z_samples, roll_samples, pitch_samples, yaw_samples,
                 fov_samples=90, aperture_samples=120, subject_pose=None, proximity_distance=50):
        if not isinstance(x_samples, collections.Iterable):
            x_samples = (x_samples,)
        if not isinstance(y_samples, collections.Iterable):
            y_samples = (y_samples,)
        if not isinstance(z_samples, collections.Iterable):
            z_samples = (z_samples,)
        if not isinstance(roll_samples, collections.Iterable):
            roll_samples = (roll_samples,)
        if not isinstance(pitch_samples, collections.Iterable):
            pitch_samples = (pitch_samples,)
        if not isinstance(yaw_samples, collections.Iterable):
            yaw_samples = (yaw_samples,)
        if not isinstance(fov_samples, collections.Iterable):
            fov_samples = (fov_samples,)
        if not isinstance(aperture_samples, collections.Iterable):
            aperture_samples = (aperture_samples,)
        self._samples = list(itertools.product(
            fov_samples, aperture_samples, x_samples, y_samples, z_samples, roll_samples, pitch_samples, yaw_samples
        ))

        # Filter samples by subject visibility
        if subject_pose is not None:
            self._samples = [(fov, aperture, x, y, z, roll, pitch, yaw)
                             for fov, aperture, x, y, z, roll, pitch, yaw in self._samples
                             if can_see_from_pose(pose=tf.Transform(location=(x, y, z), rotation=(roll, pitch, yaw)),
                                                  fov=fov,
                                                  subject_pose=subject_pose,
                                                  proximity_threshold=proximity_distance)]

        # Track if fov or aperture change, to reduce requests to the simulator
        self._last_fov = None
        self._last_aperture = None

        self._current_index = None
        self._next_settings = None
        self._settings_iterator = None
        self._simulator = None

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        if 0 <= item < len(self):
            return self.get(item)

    @property
    def sequence_type(self):
        """
        Get the kind of image sequence produced by this controller.
        :return: ImageSequenceType.NON_SEQUENTIAL
        """
        return core.sequence_type.ImageSequenceType.NON_SEQUENTIAL

    def supports_random_access(self):
        """
        True iff we can randomly access the images in this source by index.
        This controller does support random access
        :return:
        """
        return True

    @property
    def is_depth_available(self):
        """
        Can this image source produce depth images.
        Determined by the underlying simulator
        :return:
        """
        return self._simulator is not None and self._simulator.is_depth_available

    @property
    def is_per_pixel_labels_available(self):
        """
        Do images from this image source include object labels.
        Determined by the underlying simulator.
        :return: True if this image source can produce object labels for each image
        """
        return self._simulator is not None and self._simulator.is_per_pixel_labels_available

    @property
    def is_labels_available(self):
        """
        Do images from this source include object bounding boxes and simple labels in their metadata.
        Determined by the underlying simulator.
        :return: True iff the image metadata includes bounding boxes
        """
        return self._simulator is not None and self._simulator.is_labels_available

    @property
    def is_normals_available(self):
        """
        Do images from this image source include world normals.
        Determined by the underlying simulator.
        :return: True if images have world normals associated with them
        """
        return self._simulator is not None and self._simulator.is_normals_available

    @property
    def is_stereo_available(self):
        """
        Can this image source produce stereo images.
        Some algorithms only run with stereo images.
        Determined by the underlying simulator.
        :return:
        """
        return self._simulator is not None and self._simulator.is_stereo_available

    @property
    def is_stored_in_database(self):
        """
        Do this images from this source come from the database.
        Since they come from a simulator, it doesn't seem likely.
        :return:
        """
        return self._simulator is not None and self._simulator.is_stored_in_database

    def get_camera_intrinsics(self):
        """
        Get the camera intrinsics from the simulator
        :return:
        """
        return self._simulator.get_camera_intrinsics() if self._simulator is not None else None, (0, 0)

    def get_stereo_baseline(self):
        """
        Get the stereo baseline from the simulator if it is in stereo mode
        :return:
        """
        return self._simulator.get_stereo_baseline() if self._simulator is not None else None

    def begin(self):
        """
        Start producing images.
        This will trigger any necessary startup code,
        and will allow get_next_image to be called.
        Return False if there is a problem starting the source.
        :return: True iff we have successfully started iteration
        """
        if self._simulator is None:
            return False
        self._simulator.begin()
        self._current_index = 0

    def get(self, index):
        """
        If this image source supports random access, get an image by element.
        The valid indexes should be integers in the range 0 <= index < len(image_source)
        If it does not, always return None.
        Unlike get_next_image, this does not return the index or timestamp, since that has to be provided
        :param index: The index of the image to get
        :return: An image object, or None if the index is out of range.
        """
        if self._simulator is not None and self._current_index is not None and 0 <= index < len(self):
            fov, aperture, x, y, z, roll, pitch, yaw = self._samples[index]
            if self._last_fov != fov:
                self._simulator.set_field_of_view(fov)
                self._last_fov = fov
            if self._last_aperture != aperture:
                self._simulator.set_fstop(aperture)
                self._last_aperture = aperture
            self._simulator.set_camera_pose(tf.Transform(location=(x, y, z), rotation=(roll, pitch, yaw)))
            return self._simulator.get_next_image()
        return None

    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images.
        The second return value must always be

        :return: An Image object (see core.image) or None, and an index (or None)
        """
        if self._simulator is not None and self._current_index is not None:
            idx = self._current_index
            image = self.get(idx)
            self._current_index += 1
            return image, idx
        return None, None

    def is_complete(self):
        """
        Is the motion for this controller complete.
        Some controllers will produce finite motion, some will not.
        Those that do not simply always return false here.
        :return:
        """
        return self._current_index is None or self._current_index >= len(self)

    def shutdown(self):
        """
        Shut down the controller and the inner simulator
        :return:
        """
        if self._simulator is not None:
            self._simulator.shutdown()
            self._current_index = 0

    def can_control_simulator(self, simulator):
        """
        Can this controller control the given simulator.
        This one is pretty general, it needs to actually be a simulator though
        :param simulator: The simulator we may potentially control
        :return:
        """
        return simulator is not None and isinstance(simulator, simulation.simulator.Simulator)

    def set_simulator(self, simulator):
        """
        Set the simulator used by this controller
        :param simulator:
        :return:
        """
        if self.can_control_simulator(simulator):
            self._simulator = simulator


def can_see_from_pose(pose, fov, subject_pose, proximity_threshold):
    """
    Can the subject of this controller be seen from the given location.
    The sampling controller will only use poses that pass this test
    :param pose:
    :param fov:
    :param subject_pose:
    :param proximity_threshold:
    :return: True if the pose can see the subject pose
    """
    if subject_pose is None:
        # We don't have a pose for the subject, this check is useless.
        return True
    relative_subject_pose = pose.find_relative(subject_pose)
    if relative_subject_pose.location[0] < 0:
        # Object is behind the camera, we can't see it.
        return False

    distance = np.linalg.norm(relative_subject_pose.location)
    if distance < proximity_threshold:
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
