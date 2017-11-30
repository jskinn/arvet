# Copyright (c) 2017, John Skinner
import unittest
import abc
import util.dict_utils as du
import database.client
import database.entity
import database.entity_registry as reg
import database.tests.mock_database_client as mock_db_client_fac


class EntityContract(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_class(self):
        """
        Get the entity class under test
        :return:
        """
        return database.entity.Entity

    @abc.abstractmethod
    def make_instance(self, *args, **kwargs):
        """
        Make a new instance of the entity with default arguments.
        Parameters passed to this function should override the defaults.
        :param args: Forwarded to the constructor
        :param kwargs: Forwareded to the constructor
        :return: A new instance of the class under test
        """
        return None

    @abc.abstractmethod
    def assert_models_equal(self, model1, model2):
        """
        Assert that two entities are the same.
        Use the self.assertEqual methods to perform the test.
        :param model1:
        :param model2:
        :return:
        """
        pass

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Assert that two serialized models are equal.
        The default behaviour is sufficient for most cases,
        but when the serialized form contains incomparable objects (like pickled BSON),
        override this to provide better comparison.
        :param s_model1: dict
        :param s_model2: dict
        :return: 
        """
        self.assertEqual(s_model1, s_model2)

    def create_mock_db_client(self):
        """
        Create the mock database client fed to deserialize.
        The default behaviour here is sufficient if the client is not used,
        override it to create specific return values
        :return: 
        """
        self.zombie_db_client = mock_db_client_fac.create()
        return self.zombie_db_client.mock

    def assert_keys_valid(self, dictionary):
        """
        Recursively assert that all keys in this dictionary and any sub-dictionaries are valid mongodb keys.
        That is, they must be strings, and not contain the characters '.' and '$'
        :param dictionary:
        :return:
        """
        for key in dictionary.keys():
            self.assertIsInstance(key, str)
            self.assertNotIn('.', key)
            self.assertNotIn('$', key)
            if isinstance(dictionary[key], dict):
                self.assert_keys_valid(dictionary[key])

    def test_identifier(self):
        entity = self.make_instance(id_=123)
        self.assertEqual(entity.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a': 1, 'b': 2, 'c': 3}
        with self.assertRaises(TypeError):
            self.make_instance(**kwargs)

    def test_serialize_includes_fully_qualified_type(self):
        EntityClass = self.get_class()
        entity = self.make_instance(id_=123)
        s_entity = entity.serialize()
        self.assertEqual(s_entity['_type'], EntityClass.__module__ + '.' + EntityClass.__name__)

    def test_serialize_produces_valid_keys(self):
        entity = self.make_instance(id_=6224)
        s_entity = entity.serialize()
        self.assert_keys_valid(s_entity)

    def test_serialize_and_deserialize(self):
        mock_db_client = self.create_mock_db_client()
        EntityClass = self.get_class()
        entity1 = self.make_instance(id_=12345)
        s_entity1 = entity1.serialize()

        entity2 = EntityClass.deserialize(s_entity1, mock_db_client)
        s_entity2 = entity2.serialize()

        self.assert_models_equal(entity1, entity2)
        self.assert_serialized_equal(s_entity1, s_entity2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = EntityClass.deserialize(s_entity2, mock_db_client)
            s_entity2 = entity2.serialize()
            self.assert_models_equal(entity1, entity2)
            self.assert_serialized_equal(s_entity1, s_entity2)

    def test_entity_automatically_registered(self):
        EntityClass = self.get_class()
        self.assertEqual(reg.get_entity_type(EntityClass.__module__ + '.' + EntityClass.__name__), EntityClass)


class TestEntity(EntityContract, unittest.TestCase):

    def get_class(self):
        return database.entity.Entity

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'id_': 1
        })
        return database.entity.Entity(*args, **kwargs)

    def assert_models_equal(self, model1, model2):
        self.assertEqual(model1.identifier, model2.identifier)
