# Copyright (c) 2017, John Skinner
import unittest
import arvet.database.tests.database_connection as dbconn
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.core.image_source as image_source
import arvet.core.trial_result as tr
import arvet.core.tests.mock_types as mock_types
from arvet.core.sequence_type import ImageSequenceType


class MonitoredImageSource(image_source.ImageSource):
    sequence_type = ImageSequenceType.NON_SEQUENTIAL
    is_depth_available = False
    is_normals_available = False
    is_stereo_available = False
    is_labels_available = False
    is_masks_available = False
    is_stored_in_database = True
    camera_intrinsics = cam_intr.CameraIntrinsics()

    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MonitoredImageSource.instance_count += 1


class TestTrialResultDatabase(unittest.TestCase):
    system = None
    image_source = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.system.save()
        cls.image_source.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        tr.TrialResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        tr.TrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()

    def test_stores_and_loads(self):
        obj = tr.TrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            settings={'key': 'value'},
            message='Completed successfully'
        )
        obj.save()

        # Load all the entities
        all_entities = list(tr.TrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_loading_doesnt_load_image_source_unless_required(self):
        tracked_source = MonitoredImageSource()
        tracked_source.save()
        MonitoredImageSource.instance_count = 0

        obj = tr.TrialResult(
            system=self.system,
            image_source=tracked_source,
            success=True
        )
        obj.save()

        # Load all the entities
        all_entities = list(tr.TrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(MonitoredImageSource.instance_count, 0)

        # Ensure it gets lazily loaded on demand
        _ = all_entities[0].image_source
        self.assertEqual(MonitoredImageSource.instance_count, 1)

        all_entities[0].delete()
