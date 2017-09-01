import numpy as np
import transforms3d as tf3d
import core.sequence_type
import database.entity
import util.transform as tf
import simulation.simulator
import simulation.controller


class FlythroughController(simulation.controller.Controller, database.entity.Entity):
    """
    A controller that lazily flys around the level avoiding objects.
    This produces sequential image data
    """

    def __init__(self, max_speed=0.2, acceleration=0.1, max_turn_angle=np.pi / 36, avoidance_radius=1, avoidance_scale=1,
                 length=1000, seconds_per_frame=1, acceleration_noise=0.1, id_=None):
        """
        Create The controller, with some some settings
        :param max_speed: The maximum speed of the camera, in meters per frame. Default 0.2
        :param max_turn_angle: The maximum turn angle per frame, in radians. Default pi / 36 == 5 degrees
        :param avoidance_radius: The radius within which to perform obstacle avoidance
        :param length: The desired number of frames to wander for. May produce more frames if return_to_start is True.
        Default 1000.
        :param seconds_per_frame: The number of seconds passed for each frame, for scaling the timesteps. Default 1.
        """
        super().__init__(id_=id_)
        self._max_speed = float(max_speed)
        self._acceleration = float(acceleration)
        self._max_turn_angle = float(max_turn_angle)
        self._avoidance_radius = float(avoidance_radius)
        self._avoidance_scale = float(avoidance_scale)
        self._length = int(length)
        self._seconds_per_frame = float(seconds_per_frame)
        self._acceleration_noise = float(acceleration_noise)
        self._current_index = 0
        self._velocity = None
        self._simulator = None

    @property
    def sequence_type(self):
        """
        Get the kind of image sequence produced by this controller.
        :return: ImageSequenceType.NON_SEQUENTIAL
        """
        return core.sequence_type.ImageSequenceType.SEQUENTIAL

    def supports_random_access(self):
        """
        True iff we can randomly access the images in this source by index.
        This controller does not support random access
        :return:
        """
        return False

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
        return False

    def get_camera_intrinsics(self):
        """
        Get the camera intrinsics from the simulator
        :return:
        """
        return self._simulator.get_camera_intrinsics() if self._simulator is not None else None

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
        self._velocity = np.random.uniform((-1, -1, 0), (1, 1, 0), 3)
        self._velocity = self._max_speed * self._velocity / np.linalg.norm(self._velocity)

        # Set the initial facing direction to point in a random direction and random reachable location
        for _ in range(4):
            self._simulator.move_camera_to(
                tf.Transform(location=self._simulator.current_pose.location +
                                      (np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), 0)))
        self._simulator.move_camera_to(
            tf.Transform(location=self._simulator.current_pose.location +
                                  (np.random.randint(-1000, 1000), np.random.randint(-1000, 1000), 0),
                         rotation=tf3d.quaternions.axangle2quat((0, 0, 1), np.random.uniform(-np.pi, np.pi)),
                         w_first=True))

    def get(self, index):
        """
        Get an image. Since this image source doesn't support random access, return None.
        :param index:
        :return:
        """
        return None

    def get_next_image(self):
        """
        Blocking get the next image from this source.
        Parallel versions of this may add a timeout parameter.
        Returning None indicates that this image source will produce no more images.
        The second return value must always be the time

        :return: An Image object (see core.image) or None, and an index (or None)
        """
        if self._simulator is not None:
            # Choose a new camera pose
            current_pose = self._simulator.current_pose
            new_location = current_pose.location + self._seconds_per_frame * self._velocity

            # Find the direction we want to be looking, that is, the direction we're moving
            if np.any(self._velocity):
                forward = self._velocity / np.linalg.norm(self._velocity)
            else:
                forward = current_pose.forward
            up = (0, 0, 1)
            left = np.cross(up, forward)
            up = np.cross(forward, left)
            rot_mat = np.array([forward, left, up])
            new_rotation = change_orientation_toward(current_pose.rotation_quat(w_first=True),
                                                     tf3d.quaternions.mat2quat(rot_mat), self._max_turn_angle)

            # Modify the velocity
            # Accelerate in the direction we're currently moving, with some noise for random movement
            acceleration = np.zeros(3)
            while not any(acceleration):
                acceleration = (self._velocity + np.random.normal(0, self._acceleration_noise, 3))
            acceleration = self._acceleration * (acceleration / np.linalg.norm(acceleration))
            self._velocity += self._seconds_per_frame * acceleration
            # obstacle avoidance, should be mostly 0?
            self._velocity += (100 * self._avoidance_scale * self._seconds_per_frame *
                               self._simulator.get_obstacle_avoidance_force(100 * self._avoidance_radius, 100 * self._velocity))

            # Cap the max speed
            new_speed = np.linalg.norm(self._velocity)
            if new_speed > self._max_speed:
                self._velocity = self._velocity * self._max_speed / new_speed

            # Set the new position, stopping dead if we hit something
            self._simulator.move_camera_to(tf.Transform(location=new_location, rotation=new_rotation, w_first=True))
            diff = new_location - self._simulator.current_pose.location

            # Collision detection: Flip turn around and go left
            if np.dot(diff, diff) > 0.001:
                print('hit something')
                self._velocity = self._max_speed * left - self._velocity
                self._velocity = self._max_speed * self._velocity / np.linalg.norm(self._velocity)

            # Get the next image from the camera
            image, _ = self._simulator.get_next_image()
            timestamp = self._seconds_per_frame * self._current_index
            self._current_index += 1
            return image, timestamp
        return None, None

    def is_complete(self):
        """
        Is the motion for this controller complete.
        Some controllers will produce finite motion, some will not.
        Those that do not simply always return false here.
        This controller is complete when it has produced the configured number of images.
        If it must return to start, it is not complete until it subsequently returns to the start point.
        :return:
        """
        return self._current_index >= self._length

    def shutdown(self):
        """
        Shut down the controller and the inner simulator
        :return:
        """
        if self._simulator is not None:
            self._simulator.shutdown()

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

    def validate(self):
        valid = super().validate()
        return valid

    def serialize(self):
        serialized = super().serialize()
        serialized['max_speed'] = self._max_speed
        serialized['acceleration'] = self._acceleration
        serialized['max_turn_angle'] = self._max_turn_angle
        serialized['avoidance_radius'] = self._avoidance_radius
        serialized['avoidance_scale'] = self._avoidance_scale
        serialized['length'] = self._length
        serialized['seconds_per_frame'] = self._seconds_per_frame
        serialized['acceleration_noise'] = self._acceleration_noise
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'max_speed' in serialized_representation:
            kwargs['max_speed'] = serialized_representation['max_speed']
        if 'acceleration' in serialized_representation:
            kwargs['acceleration'] = serialized_representation['acceleration']
        if 'max_turn_angle' in serialized_representation:
            kwargs['max_turn_angle'] = serialized_representation['max_turn_angle']
        if 'avoidance_radius' in serialized_representation:
            kwargs['avoidance_radius'] = serialized_representation['avoidance_radius']
        if 'avoidance_scale' in serialized_representation:
            kwargs['avoidance_scale'] = serialized_representation['avoidance_scale']
        if 'length' in serialized_representation:
            kwargs['length'] = serialized_representation['length']
        if 'seconds_per_frame' in serialized_representation:
            kwargs['seconds_per_frame'] = serialized_representation['seconds_per_frame']
        if 'acceleration_noise' in serialized_representation:
            kwargs['acceleration_noise'] = serialized_representation['acceleration_noise']
        return super().deserialize(serialized_representation, db_client, **kwargs)


def change_orientation_toward(quat1, quat2, max_theta):
    """
    Change orientation quat1 toward orientation quat2, clamped to a maximum turn angle
    Based on the slerp implementation on Wikipedia: https://en.wikipedia.org/wiki/Slerp
    :param quat1: The current orientation
    :param quat2: The orientation we want
    :param max_theta: The maximum angle to change through.
    :return: The modified quaternion, same w-order as the parameters
    """
    dot = np.dot(quat1, quat2)
    if abs(dot) > 0.99995:
        # We're within 1 degree, and the math becomes unstable. Snap to the final result.
        return quat2
    if dot < 0:
        # Negative dot product means opposite 'handedness' apparently, invert the destination quat.
        # Since orientation quats are non-unique, q is the same orientation as -q
        quat2 = -1 * quat2
        dot = -1 * dot
    if dot > 1:
        dot = 1
    # The angle between quat1 and quat2
    theta = np.arccos(dot)
    if theta < max_theta:
        return quat2
    else:
        # Combination of the original vector and a purpendicular vector in the same plane as quat1 and quat2
        p = quat2 - dot * quat1
        return np.cos(max_theta) * quat1 + np.sin(max_theta) * p / np.linalg.norm(p)
