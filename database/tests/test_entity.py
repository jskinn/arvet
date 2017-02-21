from unittest import TestCase
from database.entity import Entity


class TestEntity(TestCase):

    def test_no_id(self):
        Entity()

    def test_identifier(self):
        entity = Entity(id_=123)
        self.assertEquals(entity.identifier, 123)

    def test_padded_kwargs(self):
        kwargs = {'id_': 1234, 'a':1, 'b':2, 'c': 3}
        entity = Entity(**kwargs)
        self.assertEquals(entity.identifier, 1234)

    def test_serialize_includes_type(self):
        entity = CustomEntity(id_=123)
        s_entity = entity.serialize()
        self.assertEquals(s_entity['_type'], 'CustomEntity')

    def test_serialize_and_deserialize(self):
        entity1 = Entity(id_=12345)
        s_entity1 = entity1.serialize()

        entity2 = Entity.deserialize(s_entity1)
        s_entity2 = entity2.serialize()

        self.assertEquals(entity1.identifier, entity2.identifier)
        self.assertEquals(s_entity1, s_entity2)

        for idx in range(0, 10):
            # Test that repeated serialization and deserialization does not degrade the information
            entity2 = Entity.deserialize(s_entity2)
            s_entity2 = entity2.serialize()
            self.assertEquals(entity1.identifier, entity2.identifier)
            self.assertEquals(s_entity1, s_entity2)


class CustomEntity(Entity):
    pass
