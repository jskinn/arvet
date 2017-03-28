import numpy as np
import transforms3d as tf
import util.transform as mytf


_TORAD = np.pi / 180.0


class UnrealTransform:
    """
    A special class for handling everything to do with specifying a location and orientation in Unreal specifically.
    Unreal has a bunch of painful special cases, as follows:
    - Left-handed coordinate system
    - Degrees instead of radians
    - Inconsistent direction of rotation
    This class exists to handle any particular Unreal weirdness, and keep it bottled up here
    """

    __slots__ = ['_x', '_y', '_z', '_roll', '_pitch', '_yaw']

    def __init__(self, location=None, rotation=None):
        """

        :param location: A 4x4 homogenous transformation matrix, Transform object, or and 3-indexable tuple
        :param rotation: Any 3-indexable tuple listing rotation in degrees
        """
        if isinstance(location, np.ndarray) and location.shape == (4, 4):
            location = mytf.Transform(location)
        if isinstance(location, mytf.Transform):
            location = transform_to_unreal(location)
        if isinstance(location, UnrealTransform):
            self._x = location.x
            self._y = location.y
            self._z = location.z
            self._roll = location.roll
            self._pitch = location.pitch
            self._yaw = location.yaw
        else:
            if location is not None and len(location) >= 3:
                self._x, self._y, self._z = location
            else:
                self._x = self._y = self._z = 0
            if rotation is not None and len(rotation) >= 3:
                self._roll, self._pitch, self._yaw = rotation
            else:
                self._roll = self._pitch = self._yaw = 0

    @property
    def location(self):
        """
        Get the location represented by this pose.
        :return: A numpy
        """
        return self._x, self._y, self._z

    @property
    def euler(self):
        """
        Get the Tait-Bryan angles for the rotation of this transform.
        Expressed as a numpy vector (pitch, yaw, roll),
        :return: A numpy array containing the euler angles
        """
        return self._roll, self._pitch, self._yaw

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def roll(self):
        return self._roll

    @property
    def pitch(self):
        return self._pitch

    @property
    def yaw(self):
        return self._yaw

    @property
    def forward(self):
        """
        Get the forward vector of the transform.
        That is, the positive X direction in the local coordinate frame,
        transformed to the outer coordinate frame.
        :return: The direction the pose is "facing"
        """
        return self.find_independent((1, 0, 0))

    @property
    def back(self):
        """
        Get the back vector of the transform.
        That is, the negative X direction of the local coordinate frame,
        transformed to the outer coordinate frame.
        :return: The direction backwards from the pose
        """
        return -1 * self.forward

    @property
    def up(self):
        """
        Get the up vector of the transform.
        That is
        :return: The "up" direction for this transform.
        """
        return self.find_independent((0, 0, 1))

    @property
    def down(self):
        return -1 * self.up

    @property
    def right(self):
        return self.find_independent((0, 1, 0))

    @property
    def left(self):
        return -1 * self.right

    def find_relative(self, pose):
        """
        Convert the given pose to be relative to this pose.
        This is not commutative p1.find_relative(p2) != p2.find_relative(p1)

        See Robotics: Vision and Control p 55-56 for the source of this math

        :param pose: The world pose to convert
        :return: A pose object relative to this pose
        """
        # Remember, the pose matrix gives the position in world coordinates from a local position,
        # So to find the world position, we have to reverse it
        if isinstance(pose, mytf.Transform):
            pose = transform_to_unreal(pose)
        if isinstance(pose, UnrealTransform):
            qrot = tf.taitbryan.euler2quat(_TORAD * self.yaw, _TORAD * self.pitch, _TORAD * self.roll)
            inv_rot = tf.quaternions.qinverse(qrot)
            loc = tf.quaternions.rotate_vector(np.asarray(pose.location) - np.asarray(self.location), inv_rot)

            pose_qrot = tf.taitbryan.euler2quat(_TORAD * pose.yaw, _TORAD * pose.pitch, _TORAD * pose.roll)
            pose_qrot = tf.quaternions.qmult(inv_rot, pose_qrot)
            rot = tf.taitbryan.quat2euler(pose_qrot)
            return UnrealTransform(location=loc, rotation=np.asarray(rot) / _TORAD)
        elif len(pose) >= 3:
            qrot = tf.taitbryan.euler2quat(_TORAD * self.yaw, _TORAD * self.pitch, _TORAD * self.roll)
            inv_rot = tf.quaternions.qinverse(qrot)
            return tf.quaternions.rotate_vector(np.asarray(pose) - np.asarray(self.location), inv_rot)
        else:
            raise TypeError('find_relative needs to transform a point or pose')

    def find_independent(self, pose):
        """
        Convert a pose to world coordinates.
        Remember, pose is like a stack, so the returned pose will be relative to whatever
        this pose is relative to.

        :param pose: A pose relative to this pose, as a Transform or just as a point (any length-3 indexable object)
        :return: A pose relative to whatever this pose is relative to, and independent of this pose
        """
        # REMEMBER: pre-multiplying by the transformation matrix gives the world pose from the local
        if isinstance(pose, mytf.Transform):
            pose = transform_to_unreal(pose)
        if isinstance(pose, UnrealTransform):
            qrot = tf.taitbryan.euler2quat(_TORAD * self.yaw, _TORAD * self.pitch, _TORAD * self.roll)
            pose_qrot = tf.taitbryan.euler2quat(_TORAD * pose.yaw, _TORAD * pose.pitch, _TORAD * pose.roll)

            loc = np.asarray(self.location) + tf.quaternions.rotate_vector(pose.location, qrot)
            result_qrot = tf.quaternions.qmult(qrot, pose_qrot)
            rot = tf.taitbryan.quat2euler(result_qrot)
            return UnrealTransform(location=loc, rotation=np.asarray(rot) / _TORAD)
        elif len(pose) >= 3:
            qrot = tf.taitbryan.euler2quat(_TORAD * self.yaw, _TORAD * self.pitch, _TORAD * self.roll)
            return tf.quaternions.rotate_vector(pose, qrot) + np.asarray(self.location)
        else:
            raise TypeError('find_independent needs to transform a point or pose')


def transform_to_unreal(pose):
    """
    Swap the coordinate frames from my standard coordinate frame
    to the one used by unreal
    :param pose: A point, as any 3-length indexable, or a Transform object
    :return: An UnrealTransform object.
    """
    if isinstance(pose, mytf.Transform):
        location = pose.location
        location = (location[0], location[2], location[1])
        rotation = pose.rotation_quat(w_first=True)
        rotation = (rotation[0], -rotation[1], -rotation[3], -rotation[2])
        rotation = tf.taitbryan.quat2euler(rotation)
        return UnrealTransform(location=location, rotation=np.asarray(rotation) / _TORAD)
    return pose[0], pose[2], pose[1]


def transform_from_unreal(pose):
    """
    Swap the coordinate frames from unreal coordinates
    to my standard convention
    :param pose: A point, as any 3-indexable, or a UnrealTransform object
    :return: A point or Transform object, depending on the parameter
    """
    if isinstance(pose, UnrealTransform):
        location = pose.location
        location = (location[0], location[2], location[1])
        rotation = tf.taitbryan.euler2quat(pose.yaw / _TORAD, pose.pitch / _TORAD, pose.roll / _TORAD)
        rotation = (rotation[0], -rotation[1], -rotation[3], -rotation[2])
        return mytf.Transform(location=location, rotation=rotation, w_first=True)
    return (pose[0], pose[2], pose[1])