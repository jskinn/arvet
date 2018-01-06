# Copyright (c) 2017, John Skinner
import typing
import numpy as np
import arvet.util.database_helpers as dh


class CameraIntrinsics:
    """
    An object holding all the information about the camera intrinsics for a given image.
    """

    __slots__ = ['_width', '_height', '_fx', '_fy', '_cx', '_cy', '_s', '_k1', '_k2', '_k3', '_p1', '_p2']

    def __init__(self, width: int, height: int,
                 fx: float, fy: float, cx: float, cy: float,
                 skew: float = 0.0, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0,
                 p1: float = 0.0, p2: float = 0.0):
        """
        Create a set of camera intrinsics information.
        Basic intrinsics are fractions of the image height and width.
        This loses aspect ratio information, but that is preserved in the image height and width.
        These intrinsics should therefore be consistent regardless of scaled image resolution.
        (I think that's valid?)
        :param width: The width of the image
        :param height: The height of the image
        :param fx: The x focal length, as a fraction of the image width. Should be same as fy
        :param fy: The y focal length, as a fraction of the image height. Should be same as fx
        :param cx: Principal point x coordinate, as a fraction of the image width
        :param cy: Principal point y coordinate, as a fraction of the image height
        :param skew: Pixel skew, for non-square pixels. Default 0
        :param k1: Camera radial distortion k1. Default 0
        :param k2: Camera radial distortion k2. Default 0
        :param k3: Camera radial distortion k3. Default 0
        :param p1: Camera tangential distortion p1. Default 0
        :param p2: Camera tangential distortion p2. Default 0
        """
        self._width = width
        self._height = height
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._s = skew
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2

    def __eq__(self, other: typing.Any) -> bool:
        return (isinstance(other, CameraIntrinsics) and
                self.width == other.width and
                self.height == other.height and
                self.fx == other.fx and
                self.fy == other.fy and
                self.cx == other.cx and
                self.cy == other.cy and
                self.s == other.s and
                self.k1 == other.k1 and
                self.k2 == other.k2 and
                self.k3 == other.k3 and
                self.p1 == other.p1 and
                self.p2 == other.p2)

    def __hash__(self) -> int:
        return hash((self.width, self.height,
                     self.fx, self.fy, self.cx, self.cy, self.s,
                     self.k1, self.k2, self.k3, self.p1, self.p2))

    @property
    def width(self) -> int:
        """
        The width of the image in pixels
        :return:
        """
        return self._width

    @property
    def height(self) -> int:
        """
        The height of the image in pixels
        :return:
        """
        return self._height

    @property
    def fx(self) -> float:
        """
        Get the x focal length. This is related to the width of the image
        :return:
        """
        return self._fx

    @property
    def fy(self) -> float:
        """
        Get the y focal length.
        :return:
        """
        return self._fy

    @property
    def cx(self) -> float:
        """
        Get the x coordinate of the principal point
        :return:
        """
        return self._cx

    @property
    def cy(self) -> float:
        """
        The y coordinate of the principal point
        :return:
        """
        return self._cy

    @property
    def s(self) -> float:
        """
        Image skew
        :return:
        """
        return self._s

    @property
    def k1(self) -> float:
        """
        The quadratic coefficient in a radial distortion model of the lens.
        that is x' = x(1 + k1*r^2 + k2 * r^4 + k3 *r^6), y' = y(1 + k1*r^2 + k2 * r^4 + k3 *r^6)
        See the OpenCV camera calibration
        :return:
        """
        return self._k1

    @property
    def k2(self) -> float:
        """
        The Quartic coefficient in the radial lens distortion. See k1 for details.
        :return:
        """
        return self._k2

    @property
    def k3(self) -> float:
        """
        The Sextic coefficient in the radial lens distortion. See k1 for details.
        :return:
        """
        return self._k3

    @property
    def p1(self) -> float:
        """
        The first of two tangential distortion parameters
        x' = x + (2 * p1 * x * y + p2 *(r^2 + 2x^2))
        y' = y + (2 * p2 * x * y + p1 *(r^2 + 2y^2))
        See the OpenCV camera calibration for further details
        :return:
        """
        return self._p1

    @property
    def p2(self) -> float:
        """
        The second tangential distortion parameter, see p1 for an explanation.
        :return:
        """
        return self._p2

    @property
    def horizontal_fov(self) -> float:
        """
        Get the total angle covered by the horizontal view of the camera.
        That is, given:
        |----- width -----|
        |-- cx --|
        \        |        /
         \       |       /
          \    fx|      /
           \     |     /
            \    |    /
             \ a | b /
              \  |  /
               \ | /
                \|/
        The returned angle is a + b.

        :return: The horizontal field of view, in radians
        """
        return (np.arctan2(self.cx, self.fx) +
                np.arctan2(self.width - self.cx, self.fx))

    @property
    def vertical_fov(self) -> float:
        """
        Get the total vertical angle covered by the camera. See horizontal_fov for an explanation.
        :return: The vertical field of view, in radians
        """
        return (np.arctan2(self.cy, self.fy) +
                np.arctan2(self.height - self.cy, self.fy))

    def intrinsic_matrix(self) -> np.ndarray:
        """
        Get the pinhole camera intrinsic matrix.
        Doesnt include distortion parameters, but does include skew.
        :return: A 3x3 camera matrix mapping from real world coordinates to real world.
        """
        return np.array([[self.fx, self.s, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])

    def serialize(self) -> dict:
        """
        Serialize the
        :return:
        """
        serialized = {
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'skew': self.s,
            'k1': self.k1,
            'k2': self.k2,
            'k3': self.k3,
            'p1': self.p1,
            'p2': self.p2
        }
        dh.add_schema_version(serialized, 'metadata:camera_intrinsics:CameraIntrinsics', 1)
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation: dict, **kwargs) -> 'CameraIntrinsics':
        update_schema(serialized_representation)
        if 'width' in serialized_representation:
            kwargs['width'] = serialized_representation['width']
        if 'height' in serialized_representation:
            kwargs['height'] = serialized_representation['height']
        if 'fx' in serialized_representation:
            kwargs['fx'] = serialized_representation['fx']
        if 'fy' in serialized_representation:
            kwargs['fy'] = serialized_representation['fy']
        if 'cx' in serialized_representation:
            kwargs['cx'] = serialized_representation['cx']
        if 'cy' in serialized_representation:
            kwargs['cy'] = serialized_representation['cy']
        if 'skew' in serialized_representation:
            kwargs['skew'] = serialized_representation['skew']
        if 'k1' in serialized_representation:
            kwargs['k1'] = serialized_representation['k1']
        if 'k2' in serialized_representation:
            kwargs['k2'] = serialized_representation['k2']
        if 'k3' in serialized_representation:
            kwargs['k3'] = serialized_representation['k3']
        if 'p1' in serialized_representation:
            kwargs['p1'] = serialized_representation['p1']
        if 'p2' in serialized_representation:
            kwargs['p2'] = serialized_representation['p2']
        return cls(**kwargs)


def update_schema(serialized: dict):
    """
    Update the serialized image metadata to the latest version.
    :param serialized:
    :return:
    """
    version = dh.get_schema_version(serialized, 'metadata:camera_intrinsics:CameraIntrinsics')
    if version < 1:
        # unversioned -> 1
        if 'width' in serialized:
            if serialized['fx'] < 10:   # Stored fx as a fraction of width, patch it back to what it should be
                serialized['fx'] = serialized['fx'] * serialized['width']
            if serialized['cx'] < 10:   # Stored cx as a fraction of width, patch it back to what it should be
                serialized['cx'] = serialized['cx'] * serialized['width']
        if 'height' in serialized:
            if serialized['fy'] < 10:   # Stored fy as a fraction of height, patch it back to what it should be
                serialized['fy'] = serialized['fy'] * serialized['height']
            if serialized['cy'] < 10:   # Stored cy as a fraction of height, patch it back to what it should be
                serialized['cy'] = serialized['cy'] * serialized['height']
