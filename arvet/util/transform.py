# Copyright (c) 2017, John Skinner
import numpy as np
import typing
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

    def __init__(self, location=None, rotation=None, w_first: bool = False):
        if isinstance(location, Transform):
            self._x, self._y, self._z = location.location
            self._qw, self._qx, self._qy, self._qz = location.rotation_quat(w_first=True)
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
                # I'm using the roll, pitch, yaw order of input, for consistency with Unreal
                self._qw, self._qx, self._qy, self._qz = tf.taitbryan.euler2quat(rotation[2], rotation[1], rotation[0])
            elif rotation is not None and len(rotation) >= 4:
                rotation = np.asarray(rotation, dtype=np.dtype('float'))
                if w_first:
                    self._qw, self._qx, self._qy, self._qz = robust_normalize(rotation)
                else:
                    self._qx, self._qy, self._qz, self._qw = robust_normalize(rotation)
            else:
                self._qw = 1
                self._qx = self._qy = self._qz = 0

    def __repr__(self):
        return "Transform([{x}, {y}, {z}], [{qx}, {qy}, {qz}, {qw}])".format(
            x=self._x,
            y=self._y,
            z=self._z,
            qx=self._qx,
            qy=self._qy,
            qz=self._qz,
            qw=self._qw
        )

    def __eq__(self, other):
        """
        Overridden equals method, for comparing transforms
        :param other: Another object, potentially a transform
        :return: 
        """
        if isinstance(other, Transform):
            ox, oy, oz = other.location
            oqw, oqx, oqy, oqz = other.rotation_quat(w_first=True)
            if ox == self._x and oy == self._y and oz == self._z:
                if (np.isclose(oqw, self._qw, atol=1e-16) and np.isclose(oqx, self._qx, atol=1e-16) and
                        np.isclose(oqy, self._qy, atol=1e-16) and np.isclose(oqz, self._qz, atol=1e-16)):
                    return True
                # Wasn't equal with same handedness, invert and try again because q is the same as -q for orientations
                oqw = -1 * oqw
                oqx = -1 * oqx
                oqy = -1 * oqy
                oqz = -1 * oqz
                if (np.isclose(oqw, self._qw, atol=1e-16) and np.isclose(oqx, self._qx, atol=1e-16) and
                        np.isclose(oqy, self._qy, atol=1e-16) and np.isclose(oqz, self._qz, atol=1e-16)):
                    return True
            return False
        return NotImplemented

    def __hash__(self):
        """
        Override the hash function, since we overrode equals as well.
        This lets us safely use transforms with builtin sets and dicts.
        :return: 
        """
        return hash((self._x, self._y, self._z, self._qw, self._qx, self._qy, self._qz))

    @property
    def location(self) -> np.ndarray:
        """
        Get the location represented by this pose.
        :return: A numpy
        """
        return np.array((self._x, self._y, self._z), dtype=np.dtype('f8'))

    def rotation_quat(self, w_first: bool = False) -> np.ndarray:
        """
        Get the unit quaternion representing the orientation of this pose.
        :param w_first: Is the scalar parameter the first or last element in the vector.
        :return: A 4-element numpy array that is the unit quaternion orientation
        """
        if w_first:
            return np.array((self._qw, self._qx, self._qy, self._qz), dtype=np.dtype('f8'))
        return np.array((self._qx, self._qy, self._qz, self._qw), dtype=np.dtype('f8'))

    @property
    def euler(self) -> np.ndarray:
        """
        Get the Euler angles for the rotation of this transform.
        Expressed as a numpy vector (roll, pitch, yaw),
        :return: A numpy array containing the euler angles
        """
        return tf.taitbryan.quat2euler((self._qw, self._qx, self._qy, self._qz))[::-1]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation of this transform in matrix form
        :return:
        """
        return tf.quaternions.quat2mat(self.rotation_quat(w_first=True))

    @property
    def transform_matrix(self) -> np.ndarray:
        """
        Get the homogenous transformation matrix for this pose
        :return:
        """
        return tf.affines.compose(self.location, self.rotation_matrix, np.ones(3))

    @property
    def forward(self) -> np.ndarray:
        """
        Get the forward vector of the transform.
        That is, the positive X direction in the local coordinate frame,
        transformed to the outer coordinate frame.
        :return: The direction the pose is "facing"
        """
        return tf.quaternions.rotate_vector((1, 0, 0), (self._qw, self._qx, self._qy, self._qz))

    @property
    def back(self) -> np.ndarray:
        """
        Get the back vector of the transform.
        That is, the negative X direction of the local coordinate frame,
        transformed to the outer coordinate frame.
        :return: The direction backwards from the pose
        """
        return -1 * self.forward

    @property
    def up(self) -> np.ndarray:
        """
        Get the up vector of the transform.
        That is
        :return: The "up" direction for this transform.
        """
        return tf.quaternions.rotate_vector((0, 0, 1), (self._qw, self._qx, self._qy, self._qz))

    @property
    def down(self) -> np.ndarray:
        return -1 * self.up

    @property
    def right(self) -> np.ndarray:
        return -1 * self.left

    @property
    def left(self) -> np.ndarray:
        return tf.quaternions.rotate_vector((0, 1, 0), (self._qw, self._qx, self._qy, self._qz))

    def find_relative(self, pose: typing.Union['Transform', typing.Sequence, np.ndarray]) \
            -> typing.Union['Transform', np.ndarray]:
        """
        Convert the given pose to be relative to this pose.
        This is not commutative p1.find_relative(p2) != p2.find_relative(p1)

        See Robotics: Vision and Control p 55-56 for the source of this math

        :param pose: The world pose to convert
        :return: A pose object relative to this pose, of the same type as was passed in
        """
        # Remember, the pose matrix gives the position in world coordinates from a local position,
        # So to find the world position, we have to reverse it
        if isinstance(pose, Transform):
            inv_rot = tf.quaternions.qinverse(self.rotation_quat(w_first=True))
            loc = tf.quaternions.rotate_vector(pose.location - self.location, inv_rot)
            rot = tf.quaternions.qmult(inv_rot, pose.rotation_quat(w_first=True))
            return Transform(location=loc, rotation=rot, w_first=True)
        elif len(pose) >= 3:
            return tf.quaternions.rotate_vector(pose - self.location,
                                                tf.quaternions.qinverse((self._qw, self._qx, self._qy, self._qz)))
        else:
            raise TypeError('find_relative needs to transform a point or pose')

    def find_independent(self, pose: typing.Union['Transform', typing.Sequence, np.ndarray]) \
            -> typing.Union['Transform', np.ndarray]:
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
            rot = tf.quaternions.qmult(self.rotation_quat(w_first=True),
                                       pose.rotation_quat(w_first=True))
            return Transform(location=loc, rotation=rot, w_first=True)
        elif len(pose) >= 3:
            return tf.quaternions.rotate_vector(pose, (self._qw, self._qx, self._qy, self._qz)) + self.location
        else:
            raise TypeError('find_independent needs to transform a point or pose')

    def serialize(self):
        return {
            'location': (self._x, self._y, self._z),
            'rotation': (self._qw, self._qx, self._qy, self._qz)
        }

    @classmethod
    def deserialize(cls, s_transform):
        return cls(location=s_transform['location'],
                   rotation=s_transform['rotation'],
                   w_first=True)


def robust_normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize the given vector in such a way that if we call this on it again, we get the same output.
    We care because this lets us copy Transform rotations, normalize on init, and still have exactly the same numbers,
    rather than changing them slightly each time we copy.

    Unfortunately, there are a number of vectors that we can't express with unit norm due to floating point precision
    These vectors will oscillate across 1, usually between 0.99999999999999989 and 1.0000000000000002,
    both of which are technically not 1.0. If the norm is any of these values, we just want to move on.
    :param vector:
    :return:
    """
    norm = np.linalg.norm(vector)
    loop_count = 0
    while not np.isclose(norm, 1.0, 1e-14, 1e-14) and loop_count < 10:
        vector = vector / norm
        norm = np.linalg.norm(vector)
        loop_count += 1
    return vector


def quat_angle(quat: typing.Union[typing.Sequence, np.ndarray]) -> float:
    """
    Get the angle of rotation indicated by a quaternion, independent of axis
    :param quat: A quaternion, [w, x, y, z]
    :return: The angle rotated through by the quaternion
    """
    return 2 * float(np.arccos(min(1, max(-1, quat[0]))))


def quat_diff(q1: typing.Union[typing.Sequence, np.ndarray], q2: typing.Union[typing.Sequence, np.ndarray]) -> float:
    """
    Find the angle between two quaternions
    Basically, we compose them, and derive the angle from the composition
    :param q1: A quaternion, [w, x, y, z]
    :param q2: A quaternion, [w, x, y, z]
    :return:
    """
    q1 = np.asarray(q1)
    if np.dot(q1, q2) < 0:
        # Quaternions have opposite handedness, flip q1 since it's already an ndarray
        q1 = -1 * q1
    q_inv = q1 * np.array([1.0, -1.0, -1.0, -1.0])
    q_inv = q_inv / np.linalg.norm(q_inv)

    # We only coare about the scalar component, compose only that
    z0 = q_inv[0] * q2[0] - q_inv[1] * q2[1] - q_inv[2] * q2[2] - q_inv[3] * q2[3]
    return 2 * float(np.arccos(min(1, max(-1, z0))))


def quat_mean(quaternions: typing.Sequence[typing.Union[typing.Sequence, np.ndarray]]) -> np.ndarray:
    """
    Find the mean of a bunch of quaternions
    :param quaternions:
    :return:
    """
    if len(quaternions) <= 0:
        return np.nan
    elif len(quaternions) == 1:
        # Only one quaternion, it is the average of itself
        return quaternions[0]
    elif len(quaternions) == 2 and False:
        # We have weird errors for 2 quaternions using the matrix
        # We use the closed form solution given in
        # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
        q1 = np.asarray(quaternions[0])
        q2 = np.asarray(quaternions[1])
        #if q1[0] < 0:
        #    q1 = -1 * q1
        #if q2[0] < 0:
        #    q2 = -1 * q2

        dot = np.dot(q1, q2)
        if dot < 0:
            # The vectors don't have the same handedness, invert one
            q2 = -1 * q2
            dot = -dot
        if dot == 0:
            if q1[0] > q2[0]:
                return q1
            return q2
        z = np.sqrt((q1[0] - q2[0]) * (q1[0] - q2[0]) + 4 * q1[0] * q2[0] * dot * dot)
        result1 = (q1[0] - q2[0] + z) * q1 + 2 * q2[0] * dot * q2
        result2 = 2 * q1[0] * dot * q1 + (q2[0] - q1[0] + z) * q2
        result1 /= np.linalg.norm(result1)
        result2 /= np.linalg.norm(result2)
        if not np.all(np.isclose(result1, result2)) and not np.all(np.isclose(result1, -1 * result2)):
            print("two approaches are not equal, error")
        return result1 / np.linalg.norm(result1)
    else:
        # Quaternion average from the eigenvectors of the sum matrix
        # See: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
        # We have at least 3 quaternions, make sure they're of the same handedness
        q_mat = np.asarray([
            q if np.dot(q, quaternions[0]) >= 0 else -1 * np.asarray(q)
            for q in quaternions
        ])
        product = np.dot(q_mat.T, q_mat)    # Computes sum([q * q.T for q in quaterions])
        evals, evecs = np.linalg.eig(product)
        best = -1
        result = None
        for idx in range(len(evals)):
            if evals[idx] > best:
                best = evals[idx]
                result = evecs[idx]
        if np.any(np.iscomplex(result)):
            # Mean is complex, which means the quaternions are all too close together (I think?)
            # Instead, return the Mode, the most common quaternion
            counts = [
                sum(1 for q2 in quaternions if np.array_equal(q1, q2))
                for q1 in quaternions
            ]
            best = 0
            for idx in range(len(counts)):
                if counts[idx] > best:
                    best = counts[idx]
                    result = quaternions[idx]
            print("Passing off mode as mean with {0} of {1} identical vectors".format(best, len(quaternions)))
        else:
            result = result * np.array([1, -1, -1, -1])
        return result
