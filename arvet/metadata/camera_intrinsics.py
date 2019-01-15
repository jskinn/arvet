# Copyright (c) 2017, John Skinner
import typing
import pymodm
import numpy as np


class CameraIntrinsics(pymodm.EmbeddedMongoModel):
    """
    An object holding all the information about the camera intrinsics for a given image.
    """
    width = pymodm.fields.IntegerField(required=True)
    height = pymodm.fields.IntegerField(required=True)
    fx = pymodm.fields.FloatField(required=True)
    fy = pymodm.fields.FloatField(required=True)
    cx = pymodm.fields.FloatField(required=True)
    cy = pymodm.fields.FloatField(required=True)
    s = pymodm.fields.FloatField(default=0.0)
    k1 = pymodm.fields.FloatField(default=0.0)
    k2 = pymodm.fields.FloatField(default=0.0)
    k3 = pymodm.fields.FloatField(default=0.0)
    p1 = pymodm.fields.FloatField(default=0.0)
    p2 = pymodm.fields.FloatField(default=0.0)

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
