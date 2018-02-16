# Copyright 2018 John Skinner
import numpy as np
import arvet.util.transform as tf
import arvet.simulation.simulator
import arvet.database.entity


class MockSimulator(arvet.simulation.simulator.Simulator, arvet.database.entity.Entity):

    @property
    def is_depth_available(self):
        return False

    @property
    def is_per_pixel_labels_available(self):
        return False

    @property
    def is_labels_available(self):
        return False

    @property
    def is_normals_available(self):
        return False

    @property
    def is_stereo_available(self):
        return False

    @property
    def is_stored_in_database(self):
        return True

    def get_camera_intrinsics(self):
        return None

    def begin(self):
        return True

    def get_next_image(self):
        return None, None

    def is_complete(self):
        return True

    @property
    def current_pose(self):
        return tf.Transform()

    def set_camera_pose(self, pose):
        pass

    def move_camera_to(self, pose):
        pass

    def get_obstacle_avoidance_force(self, radius=1):
        return np.array([0, 0, 0])

    @property
    def field_of_view(self):
        return 0

    def set_field_of_view(self, fov):
        pass

    @property
    def focus_distance(self):
        return 0

    def set_focus_distance(self, focus_distance):
        pass

    @property
    def fstop(self):
        return 0

    def set_fstop(self, fstop):
        pass
