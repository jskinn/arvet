# Copyright (c) 2017, John Skinner
import unittest
import pymodm
import abc
import arvet.database.tests.database_connection as dbconn
import arvet.database.pymodm_abc as pymodm_abc


class MockAbstractModel(pymodm.MongoModel, metaclass=pymodm_abc.ABCModelMeta):
    foo = pymodm.fields.FloatField(required=True)

    @property
    @abc.abstractmethod
    def bar(self):
        pass

    @abc.abstractmethod
    def required_method(self):
        pass


class ValidModel(MockAbstractModel):
    bar = pymodm.fields.BooleanField()

    def required_method(self):
        return 2


class InvalidModel(MockAbstractModel):
    baz = False


class TestImageSourceDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        MockAbstractModel._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        MockAbstractModel._mongometa.collection.drop()

    def test_stores_and_loads_valid_model(self):
        obj = ValidModel(foo=3.2, bar=False)
        obj.save()

        # Load all the entities
        all_entities = list(MockAbstractModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_cannot_instantiate_base_model(self):
        with self.assertRaises(TypeError):
            MockAbstractModel(foo=1.2)

    def test_cannot_instantiate_invalid_model(self):
        with self.assertRaises(TypeError):
            InvalidModel(foo=3.8)
