from numpy import ndarray, array as nparray
from util.geometry import (numpy_vector_to_dict,
                           numpy_quarternion_to_dict,
                           dict_vector_to_np_array,
                           dict_quaternion_to_np_array)


class Trajectory:
    """
    This class is a Trajectory, which at this point basically means a list of TrajectoryPoints.
    The key difference is that Trajectories should be immutable,
    don't go adding points to one after it has been created.
    """
    def __init__(self, points):
        self._points = points

    def __len__(self):
        return len(self._points)

    def __iter__(self):
        for point in self._points:
            yield point

    def get(self, index):
        return self._points[index]


class TrajectoryPoint:
    """
    The trajectory of an object through a video, usually the camera.
    Each trajectory consists of a timestamp, a location, and an orientation.
    location and orientation are numpy arrays, a 3D vector and 4D quaternion respectively.
    """
    def __init__(self, timestamp, location, orientation):
        self._timestamp = timestamp
        if isinstance(location, ndarray):
            if location.shape == (3,):
                self._location = location
            elif location.shape == (3, 1) or location.shape == (1, 3):
                self._location = location.reshape((3,))
        else:
            self._location = nparray([0, 0, 0])

        if isinstance(orientation, ndarray):
            if orientation.shape == (4,):
                self._orientation = orientation
            elif orientation.shape == (4, 1) or orientation.shape == (1, 4):
                self._orientation = orientation.reshape((4,))
        else:
            self._orientation = nparray([0, 0, 0, 1])

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        if timestamp > 0:
            self._timestamp = timestamp

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        if isinstance(location, ndarray):
            self._location = location

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        if isinstance(orientation, ndarray):
            self._orientation = orientation


class TrajectoryBuilder:
    """
    This is a builder class for Trajectories, so that the trajectory object itself can be immutable
    """
    def __init__(self):
        self._points = []

    def __len__(self):
        return len(self._points)

    def add_point(self, trajectory_point):
        self._points.append(trajectory_point)

    def add_arguments(self, timestamp, lx, ly, lz, qx, qy, qz, qw):
        self._points.append(TrajectoryPoint(timestamp, nparray([lx, ly, lz]), nparray([qx, qy, qz, qw])))

    def add_numpy(self, timestamp, location, orientation):
        self._points.append(TrajectoryPoint(timestamp, location, orientation))

    def build(self):
        return Trajectory(self._points)


def tuples_to_trajectory(tuple_trajectory):
    builder = TrajectoryBuilder()
    for point in tuple_trajectory:
        builder.add_arguments(point[0],  # Timestamp
                              point[1], point[2], point[3],  # location x,y,z
                              point[4], point[5], point[6], point[7])  # orientation x,y,z,w
    return builder.build()


def trajectory_to_tuples(trajectory):
    tuple_trajectory = []
    for point in trajectory:
        tuple_trajectory.append((point.timestamp,
                                 point.location[0],
                                 point.location[1],
                                 point.location[2],
                                 point.orientation[0],
                                 point.orientation[1],
                                 point.orientation[2],
                                 point.orientation[3]))
    return tuple_trajectory


def serialize_trajectory(trajectory):
    return [{
        'timestamp': trajectory_point.timestamp,
        'location': numpy_vector_to_dict(trajectory_point.location),
        'orientation': numpy_quarternion_to_dict(trajectory_point.orientation)
    } for trajectory_point in trajectory]


def deserialize_trajectory(s_trajectory):
    builder = TrajectoryBuilder()
    for s_point in s_trajectory:
        builder.add_numpy(s_point['timestamp'],
                          dict_vector_to_np_array(s_point['location']),
                          dict_quaternion_to_np_array(s_point['orientation']))
    return builder.build()
