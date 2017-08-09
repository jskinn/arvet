import unittest
import numpy as np
import pickle
import database.tests.test_entity
import util.dict_utils as du
import util.transform as tf
import core.sequence_type
import trials.slam.visual_slam as vs
import trials.slam.tracking_state as ts


class TestSLAMTrialResult(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return vs.SLAMTrialResult

    def make_instance(self, *args, **kwargs):
        states = [ts.TrackingState.NOT_INITIALIZED, ts.TrackingState.OK, ts.TrackingState.LOST]
        kwargs = du.defaults(kwargs, {
            'system_id': np.random.randint(10, 20),
            'trajectory': {
                np.random.uniform(0, 600): tf.Transform(location=np.random.uniform(-1000, 1000, 3),
                                                        rotation=np.random.uniform(0, 1, 4))
                for _ in range(100)
            },
            'ground_truth_trajectory': {
                np.random.uniform(0, 600): tf.Transform(location=np.random.uniform(-1000, 1000, 3),
                                                        rotation=np.random.uniform(0, 1, 4))
                for _ in range(100)
            },
            'tracking_stats': {
                np.random.uniform(0, 600): states[np.random.randint(0, len(states))]
                for _ in range(100)
            },
            'sequence_type': core.sequence_type.ImageSequenceType.SEQUENTIAL,
            'system_settings': {
                'a': np.random.randint(20, 30)
            }
        })
        return vs.SLAMTrialResult(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param trial_result1: 
        :param trial_result2: 
        :return:
        """
        if not isinstance(trial_result1, vs.SLAMTrialResult) or not isinstance(trial_result2, vs.SLAMTrialResult):
            self.fail('object was not a SLAMTrialResult')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1.system_id, trial_result2.system_id)
        self.assertEqual(trial_result1.success, trial_result2.success)
        self._assertTrajectoryEqual(trial_result1.trajectory, trial_result2.trajectory)
        self._assertTrajectoryEqual(trial_result1.ground_truth_trajectory, trial_result2.ground_truth_trajectory)
        self.assertEqual(trial_result1.tracking_stats, trial_result2.tracking_stats)

    def assert_serialized_equal(self, s_model1, s_model2):
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if (key is not 'ground_truth_trajectory' and
                    key is not 'trajectory' and
                    key is not 'tracking_stats'):
                self.assertEqual(s_model1[key], s_model2[key])

        traj1 = pickle.loads(s_model1['trajectory'])
        traj2 = pickle.loads(s_model2['trajectory'])
        self._assertTrajectoryEqual(traj1, traj2)

        traj1 = pickle.loads(s_model1['ground_truth_trajectory'])
        traj2 = pickle.loads(s_model2['ground_truth_trajectory'])
        self._assertTrajectoryEqual(traj1, traj2)

        stats1 = pickle.loads(s_model1['tracking_stats'])
        stats2 = pickle.loads(s_model2['tracking_stats'])
        self.assertEqual(stats1, stats2)

    def _assertTrajectoryEqual(self, traj1, traj2):
        self.assertEqual(list(traj1.keys()).sort(), list(traj2.keys()).sort())
        for time in traj1.keys():
            self.assertTrue(np.array_equal(traj1[time].location, traj2[time].location),
                            "Locations are not equal")
            self.assertTrue(np.array_equal(traj1[time].rotation_quat(w_first=True),
                                           traj2[time].rotation_quat(w_first=True)),
                            "Rotations {0} and {1} are not equal".format(tuple(traj1[time].rotation_quat(w_first=True)),
                                                                         tuple(traj2[time].rotation_quat(w_first=True))))
