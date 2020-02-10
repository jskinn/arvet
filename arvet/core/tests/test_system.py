# Copyright (c) 2017, John Skinner
import os
import unittest
import unittest.mock as mock
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.core.image import Image
from arvet.core.system import VisionSystem
from arvet.core.tests.mock_types import MockSystem, make_image


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

    def test_preload_image_data_loads_pixels(self):
        # Mock the image manager
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        image_manager.get_image = mock.Mock(wraps=image_manager.get_image)
        im_manager.set_image_manager(image_manager)

        # Make an image, and then let it go out of scope, so the data is not in memory
        image_id = make_and_store_image()

        self.assertFalse(image_manager.get_image.called)
        image = Image.objects.get({'_id': image_id})
        MockSystem.preload_image_data(image)
        self.assertTrue(image_manager.get_image.called)

        # Clean up
        im_manager.set_image_manager(None)
        Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)


def make_and_store_image():
    """
    Make an image, and then let it go out of scope, so it needs to be loaded.
    :return:
    """
    image = make_image()
    image.save()
    return image.pk
