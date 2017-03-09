import unittest
import database.entity as e
import database.entity_registry as reg


class TestEntity(unittest.TestCase):

    def test_no_id(self):
        e.Entity()

    def test_identifier(self):
        entity = e.Entity(id_=123)
        self.assertEquals(entity.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a':1, 'b':2, 'c': 3}
        with self.assertRaises(TypeError):
            e.Entity(**kwargs)

    def test_serialize_includes_type(self):
        entity = CustomEntity(id_=123)
        s_entity = entity.serialize()
        self.assertEquals(s_entity['_type'], 'CustomEntity')

    def test_serialize_and_deserialize(self):
        entity1 = e.Entity(id_=12345)
        s_entity1 = entity1.serialize()

        entity2 = e.Entity.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assertEquals(entity1.identifier, entity2.identifier)
        self.assertEquals(s_entity1, s_entity2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = e.Entity.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assertEquals(entity1.identifier, entity2.identifier)
            self.assertEquals(s_entity1, s_entity2)

    def test_entity_automatically_registered(self):
        self.assertEquals(reg.get_entity_type('CustomEntity'), CustomEntity)


class CustomEntity(e.Entity):
    pass
