import numpy as np


class CameraIntrinsics:

    __slots__ = ['_fx', '_fy', '_cx', '_cy', '_s', '_k1', '_k2', '_k3', '_p1', '_p2']

    def __init__(self, fx, fy, cx, cy, skew=0, k1=0, k2=0, k3=0, p1=0, p2=0):
        """
        Create a set of camera intrinsics information.
        Basic intrinsics are fractions of the image height and width.
        This loses aspect ratio information, but that is preserved in the image height and width.
        These intrinsics should therefore be consistent regardless of scaled image resolution.
        (I think that's valid?)
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

    def __eq__(self, other):
        return (isinstance(other, CameraIntrinsics) and
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

    def __hash__(self):
        return hash((self.fx, self.fy, self.cx, self.cy, self.s,
                     self.k1, self.k2, self.k3, self.p1, self.p2))

    @property
    def fx(self):
        """
        Get the x focal length.
        Multiply this by the width of the image to get the traditional focal length
        :return:
        """
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def s(self):
        return self._s

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def k3(self):
        return self._k3

    @property
    def p1(self):
        return self._p1

    @property
    def p2(self):
        return self._p2

    def intrinsic_matrix(self):
        """
        Get the pinhole camera intrinsic matrix.
        Doesnt include distortion parameters, but does include skew.
        :return:
        """
        return np.array([[self.fx, self.s, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])

    def serialize(self):
        return {
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

    @classmethod
    def deserialize(cls, serialized_representation, **kwargs):
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
