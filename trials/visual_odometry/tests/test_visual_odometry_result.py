import unittest
import bson.objectid
import util.dict_utils as du
import database.tests.test_entity
import util.transform as tf
import core.sequence_type
import trials.visual_odometry.visual_odometry_result as vo_res


class TestLibVisO(database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return vo_res.VisualOdometryResult

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'system_id': bson.objectid.ObjectId(),
            'sequence_type': core.sequence_type.ImageSequenceType.SEQUENTIAL,
            'system_settings': {'a': 1051},
            'frame_deltas': {
                0.3333: tf.Transform((0.1, 0.01, -0.01), (-0.01, 0.06, 1.001)),
                0.6666: tf.Transform((0.92, 0.12, 0.02), (-0.1, 0.01, 0.12)),
                1.0: tf.Transform((0.4, 0.03, -0.03), (0.03, -0.12, 0.772)),
                1.3333: tf.Transform((0.84, -0.02, 0.09), (0.013, 0.28, 0.962)),
                1.6666: tf.Transform((0.186, -0.014, -0.26), (0.7, -0.37, 0.9)),
                2.0: tf.Transform((0.37, 0.38, 0.07), (0.38, -0.27, 0.786))
            },
            'ground_truth_trajectory': {
                0.3333: tf.Transform((0.1, 0.01, -0.01), (-0.01, 0.06, 1.001)),
                0.6666: tf.Transform((25, 162, 26), (-0.1, 0.01, -0.12)),
                1.0: tf.Transform((26, 67, 9), (0.03, -0.12, 0.572)),
                1.3333: tf.Transform((82, 3, 78), (0.13, 0.25, 0.666)),
                1.6666: tf.Transform((9, 78, 6), (0.27, -0.7, 0.2)),
                2.0: tf.Transform((22, 89, 2), (0.7, -0.26, 0.87))
            }
        })
        return vo_res.VisualOdometryResult(*args, **kwargs)

    def assert_models_equal(self, system1, system2):
        """
        Helper to assert that two viso systems are equal
        :param system1:
        :param system2:
        :return:
        """
        if (not isinstance(system1, vo_res.VisualOdometryResult) or
                not isinstance(system2, vo_res.VisualOdometryResult)):
            self.fail('object was not a VisualOdometryResult')
        self.assertEqual(system1.identifier, system2.identifier)
        self.assertEqual(system1.success, system2.success)
        self.assertEqual(system1.sequence_type, system2.sequence_type)
        self.assertEqual(system1.settings, system2.settings)
        self.assertEqual(system1.frame_deltas, system2.frame_deltas)
        self.assertEqual(system1.ground_truth_trajectory, system2.ground_truth_trajectory)
