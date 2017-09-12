#Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import database.tests.test_entity
import util.dict_utils as du
import simulation.controllers.flythrough_controller as fly


class TestFlythroughController(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return fly.FlythroughController

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'max_speed': np.random.uniform(0, 10),
            'max_turn_angle': np.random.uniform(0, np.pi),
            'avoidance_radius': np.random.uniform(0, 10),
            'avoidance_scale': np.random.uniform(0, 10),
            'length': np.random.randint(0, 10000),
            'seconds_per_frame':np.random.uniform(0, 1)
        })
        return fly.FlythroughController(*args, **kwargs)

    def assert_models_equal(self, controller1, controller2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param controller1:
        :param controller2:
        :return:
        """
        if (not isinstance(controller1, fly.FlythroughController) or
                not isinstance(controller2, fly.FlythroughController)):
            self.fail('object was not a ORBSLAM2')
        self.assertEqual(controller1.identifier, controller2.identifier)
        self.assertEqual(controller1._max_speed, controller2._max_speed)
        self.assertEqual(controller1._max_turn_angle, controller2._max_turn_angle)
        self.assertEqual(controller1._avoidance_radius, controller2._avoidance_radius)
        self.assertEqual(controller1._avoidance_scale, controller2._avoidance_scale)
        self.assertEqual(controller1._length, controller2._length)
        self.assertEqual(controller1._seconds_per_frame, controller2._seconds_per_frame)
