# Copyright (c) 2017, John Skinner
import unittest
import arvet.database.tests.database_connection as dbconn
from arvet.core.system import VisionSystem
from arvet.core.tests.mock_types import MockSystem


class TestVisionSystemDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        VisionSystem._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        VisionSystem._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = MockSystem()
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_get_instance_returns_the_same_instance(self):
        system1 = MockSystem.get_instance()
        system2 = MockSystem.get_instance()
        self.assertIsNone(system1.identifier)
        self.assertIsNone(system2.identifier)

        system1.save()
        system3 = MockSystem.get_instance()
        self.assertIsNotNone(system1.identifier)
        self.assertIsNotNone(system3.identifier)
        self.assertEqual(system1.identifier, system3.identifier)
