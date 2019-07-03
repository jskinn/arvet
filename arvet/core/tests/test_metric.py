# Copyright (c) 2017, John Skinner
import unittest
from pymodm.errors import ValidationError
import arvet.database.tests.database_connection as dbconn
import arvet.core.metric as mtr
import arvet.core.tests.mock_types as mock_types


class TestMetricDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        mtr.Metric._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mtr.Metric._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = mock_types.MockMetric()
        obj.save()

        # Load all the entities
        all_entities = list(mtr.Metric.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_get_instance_returns_the_same_instance(self):
        metric1 = mock_types.MockMetric.get_instance()
        metric2 = mock_types.MockMetric.get_instance()
        self.assertIsNone(metric1.identifier)
        self.assertIsNone(metric2.identifier)

        metric1.save()
        metric3 = mock_types.MockMetric.get_instance()
        self.assertIsNotNone(metric1.identifier)
        self.assertIsNotNone(metric3.identifier)
        self.assertEqual(metric1.identifier, metric3.identifier)


class TestMetricResultDatabase(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    trial_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.metric = mock_types.MockMetric()
        cls.system.save()
        cls.image_source.save()
        cls.metric.save()

        cls.trial_result = mock_types.MockTrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        mtr.MetricResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mtr.MetricResult._mongometa.collection.drop()
        mock_types.MockTrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = mock_types.MockMetricResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            message='Completed successfully'
        )
        obj.save()

        # Load all the entities
        all_entities = list(mtr.MetricResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        # missing metric
        obj = mock_types.MockMetricResult(
            trial_results=[self.trial_result],
            success=True,
            message='Completed successfully'
        )
        with self.assertRaises(ValidationError):
            obj.save()

        # Missing trial results
        obj = mock_types.MockMetricResult(
            metric=self.metric,
            success=True,
            message='Completed successfully'
        )
        with self.assertRaises(ValidationError):
            obj.save()

        # empty trial results
        obj = mock_types.MockMetricResult(
            metric=self.metric,
            trial_results=[],
            success=True,
            message='Completed successfully'
        )
        with self.assertRaises(ValidationError):
            obj.save()

        # missing success
        obj = mock_types.MockMetricResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            message='Completed successfully'
        )
        with self.assertRaises(ValidationError):
            obj.save()


class TestCheckTrialCollection(unittest.TestCase):

    def test_returns_none_if_all_trials_succeed_and_have_same_image_source_and_system(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source, system=system, success=True)
            for _ in range(10)
        ]
        self.assertIsNone(mtr.check_trial_collection(group1))

    def test_returns_error_message_if_trials_have_failed(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source, system=system, success=True)
            for _ in range(10)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source, system=system, success=False)
            for _ in range(10)
        ]
        message = mtr.check_trial_collection(group1 + group2)
        self.assertIsNotNone(message)
        self.assertIn('failed', message)

    def test_returns_error_message_if_trials_have_different_systems(self):
        system1 = mock_types.MockSystem()
        system2 = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source, system=system1, success=True)
            for _ in range(10)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source, system=system2, success=True)
            for _ in range(10)
        ]
        message = mtr.check_trial_collection(group1 + group2)
        self.assertIsNotNone(message)
        self.assertIn('system', message)

    def test_returns_error_message_if_trials_have_different_image_sources(self):
        system = mock_types.MockSystem()
        image_source1 = mock_types.MockImageSource()
        image_source2 = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source1, system=system, success=True)
            for _ in range(10)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source2, system=system, success=True)
            for _ in range(10)
        ]
        message = mtr.check_trial_collection(group1 + group2)
        self.assertIsNotNone(message)
        self.assertIn('image source', message)

    def test_returns_different_error_messages_if_systems_or_image_sources_are_different(self):
        system1 = mock_types.MockSystem()
        system2 = mock_types.MockSystem()
        image_source1 = mock_types.MockImageSource()
        image_source2 = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source1, system=system1, success=True)
            for _ in range(10)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source2, system=system1, success=True)
            for _ in range(10)
        ]
        group3 = [
            mock_types.MockTrialResult(image_source=image_source1, system=system2, success=True)
            for _ in range(10)
        ]

        different_image_source_msg = mtr.check_trial_collection(group1 + group2)
        self.assertIn('image source', different_image_source_msg)

        different_system_msg = mtr.check_trial_collection(group1 + group3)
        self.assertNotEqual(different_system_msg, different_image_source_msg)
        self.assertIn('system', different_system_msg)
