import unittest
import numpy as np
import pickle
import core.sequence_type
import database.tests.test_entity
import util.dict_utils as du
import util.transform as tf
import simulation.controllers.trajectory_follow_controller as follow


class TestTrajectoryFollowController(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return follow.TrajectoryFollowController

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'trajectory': {i + np.random.uniform(-0.2, 0.2): tf.Transform(np.random.uniform(-1000, 1000, 3),
                                                                          np.random.uniform(-1, 1, 4))
                           for i in range(1000)},
            'sequence_type': core.sequence_type.ImageSequenceType(np.random.randint(0, 2))
        })
        return follow.TrajectoryFollowController(*args, **kwargs)

    def assert_models_equal(self, controller1, controller2):
        """
        Helper to assert that two controllers are equal
        :param controller1:
        :param controller2:
        :return:
        """
        if (not isinstance(controller1, follow.TrajectoryFollowController) or
                not isinstance(controller2, follow.TrajectoryFollowController)):
            self.fail('object was not a TrajectoryFollowController')
        self.assertEqual(controller1.identifier, controller2.identifier)
        self.assertEqual(controller1.sequence_type, controller2.sequence_type)
        self._assertTrajectoryEqual(controller1._trajectory, controller2._trajectory)

    def assert_serialized_equal(self, s_model1, s_model2):
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        self.assertEqual(s_model1['sequence_type'], s_model2['sequence_type'])
        traj1 = pickle.loads(s_model1['trajectory'])
        traj2 = pickle.loads(s_model2['trajectory'])
        self._assertTrajectoryEqual(traj1, traj2)

    def _assertTrajectoryEqual(self, traj1, traj2):
        self.assertEqual(list(traj1.keys()).sort(), list(traj2.keys()).sort())
        for time in traj1.keys():
            self.assertTrue(np.array_equal(traj1[time].location, traj2[time].location),
                            "Locations are not equal")
            self.assertTrue(np.array_equal(traj1[time].rotation_quat(w_first=True),
                                           traj2[time].rotation_quat(w_first=True)),
                            "Rotations {0} and {1} are not equal".format(tuple(traj1[time].rotation_quat(w_first=True)),
                                                                         tuple(traj2[time].rotation_quat(w_first=True))))
