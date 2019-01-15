# Copyright (c) 2017, John Skinner
import unittest
import arvet.database.tests.database_connection as dbconn
import arvet.core.trial_result as tr
import arvet.core.metric as mtr
import arvet.core.tests.mock_types as mock_types


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

        cls.trial_result = tr.TrialResult(image_source=cls.image_source, system=cls.system, success=True)
        cls.trial_result.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        mtr.MetricResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mtr.MetricResult._mongometa.collection.drop()
        tr.TrialResult._mongometa.collection.drop()
        mock_types.MockMetric._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = mtr.MetricResult(
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
