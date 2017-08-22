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
            'orbslam_systems': [oid.ObjectId(), oid.ObjectId(), oid.ObjectId()],
            'benchmark_rpe': oid.ObjectId(),
            'benchmark_trajectory_drift': oid.ObjectId(),
            'datasets': [oid.ObjectId(), oid.ObjectId(), oid.ObjectId()],
            'base_datasets': [oid.ObjectId(), oid.ObjectId(), oid.ObjectId()],
            'trial_list': [(oid.ObjectId(), oid.ObjectId(), oid.ObjectId())],
            'result_list': [(oid.ObjectId(), oid.ObjectId(), oid.ObjectId(), oid.ObjectId())]
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
        self.assertEqual(experiment1._orbslam_systems, experiment2._orbslam_systems)
        self.assertEqual(experiment1._benchmark_rpe, experiment2._benchmark_rpe)
        self.assertEqual(experiment1._benchmark_trajectory_drift, experiment2._benchmark_trajectory_drift)
        self.assertEqual(experiment1._datasets, experiment2._datasets)
        self.assertEqual(experiment1._base_datasets, experiment2._base_datasets)
        self.assertEqual(experiment1._trial_list, experiment2._trial_list)
        self.assertEqual(experiment1._result_list, experiment2._result_list)

    def test_constructor_works_with_minimal_arguments(self):
        vse.VisualSlamExperiment()
