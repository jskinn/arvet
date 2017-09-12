#Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import database.entity_registry as reg


class MockEntity:
    pass
reg.register_entity(MockEntity)


# Make some additional entities for testing, which further clutter our global namespaces...
mock_class_1 = mock.MagicMock()
mock_class_1.__module__ = 'mock_module.test1'
mock_class_1.__name__ = 'MagicMockEntity'
reg.register_entity(mock_class_1)
mock_class_2 = mock.MagicMock()
mock_class_2.__module__ = 'mock_module.test2'
mock_class_2.__name__ = 'MagicMockEntity'
reg.register_entity(mock_class_2)


class TestEntityRegistry(unittest.TestCase):

    def test_can_retrieve_entity_class_by_fully_qualified_name(self):
        name = MockEntity.__module__ + '.' + MockEntity.__name__
        self.assertEqual(MockEntity, reg.get_entity_type(name))

    def test_can_retrieve_unique_entity_class_by_class_name_if_unique(self):
        self.assertEqual(MockEntity, reg.get_entity_type(MockEntity.__name__))

    def test_cannot_retrieve_non_unique_entity_class_by_name(self):
        self.assertIsNone(reg.get_entity_type('MagicMockEntity'))

    def test_find_potential_entity_classes_returns_list_of_matching_qualified_class_names(self):
        self.assertEqual({'mock_module.test1.MagicMockEntity',
                          'mock_module.test2.MagicMockEntity'},
                         set(reg.find_potential_entity_classes('MagicMockEntity')))
