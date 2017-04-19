import numpy as np


def dict_vector_to_np_array(data):
    return np.array([data['X'], data['Y'], data['Z']])


def dict_quaternion_to_np_array(data):
    return np.array([data['X'], data['Y'], data['Z'], data['W']])


def numpy_vector_to_dict(vector):
    return {'X': vector[0], 'Y': vector[1], 'Z': vector[2]}


def numpy_quarternion_to_dict(quarternion):
    return {'X': quarternion[0], 'Y': quarternion[1], 'Z': quarternion[2], 'W': quarternion[3]}
