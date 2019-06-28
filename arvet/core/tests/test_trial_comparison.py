# Copyright (c) 2017, John Skinner
import unittest
from pymodm.errors import ValidationError
import arvet.database.tests.database_connection as dbconn
import arvet.core.trial_comparison as mtr_comp
import arvet.core.tests.mock_types as mock_types


class TestTrialComparisonMetricDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        mtr_comp.TrialComparisonMetric._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mtr_comp.TrialComparisonMetric._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = mock_types.MockTrialComparisonMetric()
        obj.save()

        # Load all the entities
        all_entities = list(mtr_comp.TrialComparisonMetric.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_get_instance_returns_the_same_instance(self):
        system1 = mock_types.MockTrialComparisonMetric.get_instance()
        system2 = mock_types.MockTrialComparisonMetric.get_instance()
        self.assertIsNone(system1.identifier)
        self.assertIsNone(system2.identifier)

        system1.save()
        system3 = mock_types.MockTrialComparisonMetric.get_instance()
        self.assertIsNotNone(system1.identifier)
        self.assertIsNotNone(system3.identifier)
        self.assertEqual(system1.identifier, system3.identifier)


class TestTrialComparisonMetricResultDatabase(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    trial_result_1 = None
    trial_result_2 = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.metric = mock_types.MockTrialComparisonMetric()
        cls.system.save()
        cls.image_source.save()
        cls.metric.save()

        cls.trial_result_1 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_1.save()
        cls.trial_result_2 = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result_2.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        mtr_comp.TrialComparisonResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mtr_comp.TrialComparisonResult._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = mtr_comp.TrialComparisonResult(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            success=True,
            message='Completed successfully'
        )
        obj.save()

        # Load all the entities
        all_entities = list(mtr_comp.TrialComparisonResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        # No metric
        obj = mtr_comp.TrialComparisonResult(
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2],
            success=True
        )
        with self.assertRaises(ValidationError):
            obj.save()

        # No trial results 1
        obj = mtr_comp.TrialComparisonResult(
            metric=self.metric,
            trial_results_2=[self.trial_result_2],
            success=True
        )
        with self.assertRaises(ValidationError):
            obj.save()

        # No trial results 2
        obj = mtr_comp.TrialComparisonResult(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            success=True
        )
        with self.assertRaises(ValidationError):
            obj.save()

        # No success
        obj = mtr_comp.TrialComparisonResult(
            metric=self.metric,
            trial_results_1=[self.trial_result_1],
            trial_results_2=[self.trial_result_2]
        )
        with self.assertRaises(ValidationError):
            obj.save()
