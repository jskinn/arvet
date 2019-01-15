# Copyright (c) 2017, John Skinner
import unittest
import arvet.database.tests.database_connection as dbconn
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image_source import ImageSource
from arvet.metadata.camera_intrinsics import CameraIntrinsics


class MockImageSource(ImageSource):
    sequence_type = ImageSequenceType.NON_SEQUENTIAL
    is_depth_available = False
    is_normals_available = False
    is_stereo_available = False
    is_labels_available = False
    is_masks_available = False
    is_stored_in_database = True
    camera_intrinsics = CameraIntrinsics()


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
