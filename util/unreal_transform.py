import numpy as np
import transforms3d as tf
import util.transform as mytf


_TORAD = np.pi / 180.0
_TODEG = 180 / np.pi


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
        :param rotation: Any 3-indexable tuple listing rotation in degrees, order (roll, pitch, yaw)
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
            inv_mat = np.transpose(euler2mat(self.roll, self.pitch, self.yaw))
            pose_mat = euler2mat(pose.roll, pose.pitch, pose.yaw)

            loc = np.dot(inv_mat, np.asarray(pose.location) - np.asarray(self.location))
            rot = np.dot(inv_mat, pose_mat)
            rot = mat2euler(rot)

            quat = euler2quat(self.roll, self.pitch, self.yaw)
            inv_quat = tf.quaternions.qconjugate(quat)
            pose_quat = euler2quat(pose.roll, pose.pitch, pose.yaw)

            loc = tf.quaternions.rotate_vector(np.asarray(pose.location) - np.asarray(self.location), inv_quat)
            rot = tf.quaternions.qmult(inv_quat, pose_quat)

            return UnrealTransform(location=loc, rotation=quat2euler(rot[0], rot[1], rot[2], rot[3]))

        elif len(pose) >= 3:
            #inv_mat = np.transpose(euler2mat(self.roll, self.pitch, self.yaw))
            #return np.dot(inv_mat, np.asarray(pose) - np.asarray(self.location))
            inv_quat = tf.quaternions.qinverse(euler2quat(self.roll, self.pitch, self.yaw))
            return tf.quaternions.rotate_vector(np.asarray(pose) - np.asarray(self.location), inv_quat)
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
            quat = euler2quat(self.roll, self.pitch, self.yaw)
            pose_quat = euler2quat(pose.roll, pose.pitch, pose.yaw)

            rot = tf.quaternions.qmult(quat, pose_quat)
            loc = np.asarray(self.location) + np.asarray(tf.quaternions.rotate_vector(pose.location, quat))

            return UnrealTransform(location=loc, rotation=quat2euler(rot[0], rot[1], rot[2], rot[3]))
        elif len(pose) >= 3:
            #mat = euler2mat(self.roll, self.pitch, self.yaw)
            #return np.dot(mat, np.asarray(pose)) + np.asarray(self.location)
            quat = euler2quat(self.roll, self.pitch, self.yaw)
            return tf.quaternions.rotate_vector(np.asarray(pose), quat) + np.asarray(self.location)

        else:
            raise TypeError('find_independent needs to transform a point or pose')


def euler2mat(roll, pitch, yaw):
    """
    Create a rotation matrix for the orientation expressed by this transform.
    Copied directly from FRotationTranslationMatrix::FRotationTranslationMatrix
    in Engine/Source/Runtime/Core/Public/Math/RotationTranslationMatrix.h ln 32
    :return:
    """
    angles = _TORAD * np.array((roll, pitch, yaw))
    sr, sp, sy = np.sin(angles)
    cr, cp, cy = np.cos(angles)

    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, -(cr * sp * cy + sr * sy)],
        [cp * sy, sr * sp * sy + cr * cy, cy * sr - cr * sp * sy],
        [sp     , -sr * cp              , cr * cp]
    ])


def mat2euler(mat):
    """
    Go back from a rotation matrix to euler angles
    This is copied from FMatrix::Rotator(),
    in Engine/Source/Runtime/Core/Private/Math/UnrealMath.cpp ln 473

    :param mat:
    :return:
    """
    x_axis = mat[0, 0:3]
    y_axis = mat[1, 0:3]
    z_axis = mat[2, 0:3]

    pitch = np.arctan2(x_axis[2], np.sqrt(x_axis[0] * x_axis[0] + x_axis[1] * x_axis[1])) * _TODEG
    yaw = np.arctan2(x_axis[1], x_axis[0]) / _TORAD

    temp_mat = euler2mat(0, pitch, yaw)
    sy_axis = temp_mat[1, 0:3]
    # For some crazy reason '|' means dot product for unreal FVector
    roll = np.arctan2(np.dot(z_axis, sy_axis), np.dot(y_axis, sy_axis)) * _TODEG
    return roll, pitch, yaw


def euler2quat(roll, pitch, yaw):
    """
    Convert Unreal Euler angles to a quaternion.
    Based on FRotator::Quaternion in
    Engine/Source/Runtime/Core/Private/Math/UnrealMath.cpp ln 373
    :param roll: Roll angle
    :param pitch: Pitch angle
    :param yaw: Yaw angle
    :return: A tuple quaternion in unreal space, w first
    """
    angles = np.array([roll, pitch, yaw])
    angles = angles * _TORAD / 2
    sr, sp, sy = np.sin(angles)
    cr, cp, cy = np.cos(angles)

    x =  cr * sp * sy - sr * cp * cy
    y = -cr * sp * cy - sr * cp * sy
    z =  cr * cp * sy - sr * sp * cy
    w =  cr * cp * cy + sr * sp * sy
    return w, x, y, z


def quat2euler(w, x, y, z):
    """
    Convert a quaternion in unreal space to euler angles.
    Based on FQuat::Rotator in
    Engine/Source/Runtime/Core/Private/Math/UnrealMath.cpp ln 536
    which is in turn based on
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param w:
    :param x:
    :param y:
    :param z:
    :return:
    """
    SINGULARITY_THRESHOLD = 0.4999995

    singularity_test = z * x - w * y
    yaw_y = 2 * (w * z + x * y)
    yaw_x = 1 - 2 * (y * y + z * z)

    yaw = np.arctan2(yaw_y, yaw_x) * _TODEG
    if singularity_test < -SINGULARITY_THRESHOLD:
        pitch = -90
        roll = _clamp_axis(-yaw - 2 * np.arctan2(x, w) * _TODEG)
    elif singularity_test > SINGULARITY_THRESHOLD:
        pitch = 90
        roll = _clamp_axis(yaw - 2 * np.arctan2(x, w) * _TODEG)
    else:
        pitch = np.arcsin(2 * singularity_test) / _TORAD
        roll = np.arctan2(-2 * (w * x + y * z), (1 - 2 * (x * x + y * y))) * _TODEG
    return roll, pitch, yaw


def _clamp_axis(angle):
    angle %= 360
    if angle < -180:
        angle += 360
    return angle


def transform_to_unreal(pose):
    """
    Swap the coordinate frames from my standard coordinate frame
    to the one used by unreal
    :param pose: A point, as any 3-length indexable, or a Transform object
    :return: An UnrealTransform object.
    """
    if isinstance(pose, mytf.Transform):
        location = (pose.location[0], -pose.location[1], pose.location[2])
        rotation = pose.rotation_quat(w_first=True)
        # Invert Y axis to go to quaternion in unreal frame
        rotation = (rotation[0], rotation[1], -rotation[2], rotation[3])
        # Invert the direction of rotation since we're now in a left handed coordinate frame
        rotation = tf.quaternions.qinverse(rotation)
        #rotation = tf.taitbryan.quat2euler(rotation)
        # Change the axis order to roll, pitch, yaw in UE coordinates
        #rotation = np.array([rotation[2], rotation[1], rotation[0]])
        #return UnrealTransform(location=location, rotation=rotation * _TODEG)
        return UnrealTransform(location=location, rotation=quat2euler(rotation[0], rotation[1], rotation[2], rotation[3]))
    return pose[0], -pose[1], pose[2]


def transform_from_unreal(pose):
    """
    Swap the coordinate frames from unreal coordinates
    to my standard convention
    :param pose: A point, as any 3-indexable, or a UnrealTransform object
    :return: A point or Transform object, depending on the parameter
    """
    if isinstance(pose, UnrealTransform):
        location = (pose.location[0], -pose.location[1], pose.location[2])
        #rotation = tf.taitbryan.euler2quat(pose.yaw * _TORAD, pose.pitch * _TORAD, pose.roll * _TORAD)
        rotation = euler2quat(pose.roll, pose.pitch, pose.yaw)
        # Invert the direction of rotation to go to a right-handed coordinate frame
        rotation = tf.quaternions.qinverse(rotation)
        # Invert Y-axis to go to my coordinate frame
        rotation = (rotation[0], rotation[1], -rotation[2], rotation[3])


        return mytf.Transform(location=location, rotation=rotation, w_first=True)
    return pose[0], -pose[1], pose[2]
