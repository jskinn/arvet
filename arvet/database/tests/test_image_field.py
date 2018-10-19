import os.path
import pymodm
import numpy as np
from arvet.util.test_helpers import ExtendedTestCase
import arvet.database.tests.database_connection as dbconn
from arvet.database.image_field import ImageField
import arvet.database.image_manager as im_manager


class TestImageFieldMongoModel(pymodm.MongoModel):
    image = ImageField()


class TestImageField(ExtendedTestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model and removing the image file
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        TestImageFieldMongoModel._mongometa.collection.drop()

    def test_image_field_stores_and_loads(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Save the image
        model = TestImageFieldMongoModel()
        model.image = image
        model.save()

        # Load all the entities
        all_entities = list(TestImageFieldMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertNPEqual(all_entities[0].image, image)
        all_entities[0].delete()

    def test_delete_removes(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)
        path = image_manager.find_path_for_image(image)

        # Save the image
        model = TestImageFieldMongoModel()
        model.image = image
        model.save()
        self.assertTrue(image_manager.is_valid_path(path))

        # Delete the model
        model.delete()
        self.assertFalse(image_manager.is_valid_path(path))
