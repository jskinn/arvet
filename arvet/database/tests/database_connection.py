import os.path as pth
from pymodm.connection import connect


__connected = False


def connect_to_test_db():
    global __connected
    if not __connected:
        connect("mongodb://localhost:27017/arvet-test-db")
        __connected = True


image_file = pth.join(pth.dirname(pth.abspath(__file__)), 'test-images.hdf5')
