# Copyright (c) 2017, John Skinner
import unittest
import arvet.database.tests.database_connection as dbconn
from arvet.core.image_source import ImageSource
from arvet.core.tests.mock_types import MockImageSource


class TestImageSourceDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        ImageSource._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        ImageSource._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = MockImageSource()
        obj.save()

        # Load all the entities
        all_entities = list(ImageSource.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()
