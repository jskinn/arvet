import unittest
import unittest.mock as mock
import bson.objectid as oid
import pymongo.collection
import util.dict_utils as du
import database.client
import database.tests.test_entity as entity_test
import batch_analysis.experiment as ex
import experiments.visual_slam.visual_slam_experiment as vse


class TestVisualSlamExperiment(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return vse.VisualSlamExperiment

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'libviso_system': oid.ObjectId(),
            'orbslam_monocular_system': oid.ObjectId(),
            'orbslam_stereo_system': oid.ObjectId(),
            'orbslam_rgbd_system': oid.ObjectId(),
            'benchmark_rpe': oid.ObjectId(),
            'benchmark_trajectory_drift': oid.ObjectId()
        })
        return vse.VisualSlamExperiment(*args, **kwargs)

    def assert_models_equal(self, experiment1, experiment2):
        """
        Helper to assert that two visual slam experiments are equal
        :param experiment1: First experiment to compare
        :param experiment2: Second experiment to compare
        :return:
        """
        if (not isinstance(experiment1, vse.VisualSlamExperiment) or
                not isinstance(experiment2, vse.VisualSlamExperiment)):
            self.fail('object was not a VisualSlamExperiment')
        self.assertEqual(experiment1.identifier, experiment2.identifier)
        self.assertEqual(experiment1._libviso_system, experiment2._libviso_system)
        self.assertEqual(experiment1._orbslam_monocular, experiment2._orbslam_monocular)
        self.assertEqual(experiment1._orbslam_stereo, experiment2._orbslam_stereo)
        self.assertEqual(experiment1._orbslam_rgbd, experiment2._orbslam_rgbd)
        self.assertEqual(experiment1._benchmark_rpe, experiment2._benchmark_rpe)
        self.assertEqual(experiment1._benchmark_trajectory_drift, experiment2._benchmark_trajectory_drift)

    def test_constructor_works_with_minimal_arguments(self):
        vse.VisualSlamExperiment()
