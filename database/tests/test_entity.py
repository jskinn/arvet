import unittest
import abc
import util.dict_utils as du
import database.entity
import database.entity_registry as reg


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
        Use the self.assertEquals methods to perform the test.
        :param model1:
        :param model2:
        :return:
        """
        pass

    def test_identifier(self):
        entity = self.make_instance(id_=123)
        self.assertEquals(entity.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a': 1, 'b': 2, 'c': 3}
        with self.assertRaises(TypeError):
            self.make_instance(**kwargs)

    def test_serialize_includes_type(self):
        EntityClass = self.get_class()
        entity = self.make_instance(id_=123)
        s_entity = entity.serialize()
        self.assertEquals(s_entity['_type'], EntityClass.__name__)

    def test_serialize_and_deserialize(self):
        EntityClass = self.get_class()
        entity1 = self.make_instance(id_=12345)
        s_entity1 = entity1.serialize()

        entity2 = EntityClass.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assert_models_equal(entity1, entity2)
        self.assertEquals(s_entity1, s_entity2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = EntityClass.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assert_models_equal(entity1, entity2)
            self.assertEquals(s_entity1, s_entity2)

    def test_entity_automatically_registered(self):
        EntityClass = self.get_class()
        self.assertEquals(reg.get_entity_type(EntityClass.__name__), EntityClass)


class TestEntity(EntityContract, unittest.TestCase):

    def get_class(self):
        return database.entity.Entity

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'id_': 1
        })
        return database.entity.Entity(*args, **kwargs)

    def assert_models_equal(self, model1, model2):
        self.assertEquals(model1.identifier, model2.identifier)
