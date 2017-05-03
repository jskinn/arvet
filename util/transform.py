import numpy as np
import transforms3d as tf


######
# TODO: We need to handle transforming between coordinate frames where the axies don't point in the same direction
# for instance, Unreal's coordinates have X as forward, Y as up, and Z as right
# but an algorithm might have Z as up and Y as left, or Z as forward and X as right
# I need to establish what coordinate frame I'm using, and write code to convert everyone else's to mine.
#####

class Transform:
    """
    A class for handling everything to do with specifying a location and orientation in 3D space.
    Different systems have different conventions, you need to keep track of:
    - Handedness (left or right)
    - Which axis points in which direction
    - Order of rotations for euler angles

    I'm going to establish conventions here, and then for each system, convert to and from them.
    THESES ARE THE RULES:
    - X is forward, Y is left, Z is up
    - This is a right-handed coordinate system
    - Euler angles are applied Z, Y, X for yaw, pitch, and then roll, matching tait-bryan approach
    YOU MUST CONVERT TO THIS COORDINATE SYSTEM

    This class can handle quaternions with W either as the first or last element.
    """

    __slots__ = ['_x', '_y', '_z', '_qx', '_qy', '_qz', '_qw']

    def __init__(self, location=None, rotation=None, w_first=False):
        if isinstance(location, Transform):
            self._x = location._x
            self._y = location._y
            self._z = location._z
            self._qw = location._qw
            self._qx = location._qx
            self._qy = location._qy
            self._qz = location._qz
        elif isinstance(location, np.ndarray) and location.shape == (4, 4):
            # first parameter is a homogenous transformation matrix, turn it back into
            # location and quaternion
            trans, rot, zooms, shears = tf.affines.decompose44(location)
            self._x, self._y, self._z = trans
            self._qw, self._qx, self._qy, self._qz = tf.quaternions.mat2quat(rot)
        else:
            # Location
            if location is not None and len(location) >= 3:
                self._x, self._y, self._z = location
            else:
                self._x = self._y = self._z = 0
            # rotation
            if rotation is not None and len(rotation) == 3:
                # Rotation is euler angles, convert to quaternion
                # I'm using the tait-bryan order of operations, for consistency with Unreal
                self._qw, self._qx, self._qy, self._qz = tf.taitbryan.euler2quat(rotation[2], rotation[1], rotation[0])
            elif rotation is not None and len(rotation) >= 4:
                if w_first:
                    self._qw, self._qx, self._qy, self._qz = rotation / np.linalg.norm(rotation)
                else:
                    self._qx, self._qy, self._qz, self._qw = rotation / np.linalg.norm(rotation)
            else:
                self._qw = 1
                self._qx = self._qy = self._qz = 0

    @property
    def location(self):
        """
        Get the location represented by this pose.
        :return: A numpy
        """
        return np.array([self._x, self._y, self._z])

    def rotation_quat(self, w_first=False):
        """
        Get the unit quaternion representing the orientation of this pose.
        :param w_first: Is the scalar parameter the first or last element in the vector.
        :return: A 4-element numpy array that is the unit quaternion orientation
        """
        if w_first:
            return np.array([self._qw, self._qx, self._qy, self._qz])
        return np.array([self._qx, self._qy, self._qz, self._qw])

    @property
    def euler(self):
        """
        Get the Tait-Bryan angles for the rotation of this transform.
        Expressed as a numpy vector (pitch, yaw, roll),
        :return: A numpy array containing the euler angles
        """
        return tf.taitbryan.quat2euler((self._qw, self._qx, self._qy, self._qz))

    @property
    def transform_matrix(self):
        """
        Get the homogenous transformation matrix for this pose
        :return:
        """
        return tf.affines.compose(self.location,
                                  tf.quaternions.quat2mat(self.rotation_quat(w_first=True)),
                                  np.ones(3))

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
        return self.find_independent((0, 1, 0))

    @property
    def down(self):
        return -1 * self.up

    @property
    def right(self):
        return self.find_independent((0, 0, 1))

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
        if isinstance(pose, Transform):
            inv_rot = tf.quaternions.qinverse((self._qw, self._qx, self._qy, self._qz))
            loc = tf.quaternions.rotate_vector(pose.location - self.location, inv_rot)
            rot = tf.quaternions.qmult(inv_rot, (pose._qw, pose._qx, pose._qy, pose._qz))
            return Transform(location=loc, rotation=rot, w_first=True)
        elif len(pose) >= 3:
            return tf.quaternions.rotate_vector(pose - self.location,
                                                tf.quaternions.qinverse((self._qw, self._qx, self._qy, self._qz)))
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
        if isinstance(pose, Transform):
            loc = self.location + tf.quaternions.rotate_vector(pose.location, (self._qw, self._qx, self._qy, self._qz))
            rot = tf.quaternions.qmult((self._qw, self._qx, self._qy, self._qz),
                                       (pose._qw, pose._qx, pose._qy, pose._qz))
            return Transform(location=loc, rotation=rot, w_first=True)
        elif len(pose) >= 3:
            return tf.quaternions.rotate_vector(pose, (self._qw, self._qx, self._qy, self._qz)) + self.location
        else:
            raise TypeError('find_independent needs to transform a point or pose')


def serialize_transform(transform):
    return {
        'location': tuple(transform.location),
        'rotation': tuple(transform.rotation_quat(w_first=True))
    }


def deserialize_transform(s_transform):
    return Transform(location=tuple(s_transform['location']),
                     rotation=tuple(s_transform['rotation']),
                     w_first=True)
