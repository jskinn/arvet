import pymodm
import unittest
import arvet.util.transform as tf
import arvet.database.tests.database_connection as dbconn
from arvet.database.transform_field import TransformField


class TestTransformFieldMongoModel(pymodm.MongoModel):
    transform = TransformField()


class TestImageField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        TestTransformFieldMongoModel._mongometa.collection.drop()

    def test_image_field_stores_and_loads(self):
        pose = tf.Transform(location=[1, 2, 3], rotation=[4, 5, 6])

        # Save the model
        model = TestTransformFieldMongoModel()
        model.transform = pose
        model.save()

        # Load all the entities
        all_entities = list(TestTransformFieldMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].transform, pose)
        all_entities[0].delete()
