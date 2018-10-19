import enum
import pymodm
import unittest
import arvet.database.tests.database_connection as dbconn
from arvet.database.enum_field import EnumField


class MyEnum(enum.Enum):
    FOO = 0
    BAR = 1
    BAZ = 2


class TestEnumFieldMongoModel(pymodm.MongoModel):
    enum = EnumField(MyEnum)


class TestImageField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        TestEnumFieldMongoModel._mongometa.collection.drop()

    def test_image_field_stores_and_loads(self):

        # Save the model
        model = TestEnumFieldMongoModel()
        model.enum = MyEnum.BAZ
        model.save()

        # Load all the entities
        all_entities = list(TestEnumFieldMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].enum, MyEnum.BAZ)
        all_entities[0].delete()
