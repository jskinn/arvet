# Copyright (c) 2017, John Skinner
import typing
import pymodm
import numpy as np


class CameraIntrinsics(pymodm.EmbeddedMongoModel):
    """
    An object holding all the information about the camera intrinsics for a given image.
    """
    _width = pymodm.fields.IntegerField(required=True)
    _height = pymodm.fields.IntegerField(required=True)
    _fx = pymodm.fields.FloatField(required=True)
    _fy = pymodm.fields.FloatField(required=True)
    _cx = pymodm.fields.FloatField(required=True)
    _cy = pymodm.fields.FloatField(required=True)
    _s = pymodm.fields.FloatField(default=0.0)
    _k1 = pymodm.fields.FloatField(default=0.0)
    _k2 = pymodm.fields.FloatField(default=0.0)
    _k3 = pymodm.fields.FloatField(default=0.0)
    _p1 = pymodm.fields.FloatField(default=0.0)
    _p2 = pymodm.fields.FloatField(default=0.0)

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
