import unittest
import unittest.mock as mock
import pymodm
import arvet.database.tests.database_connection as dbconn
from .mock_base_model import TestBaseModel
from arvet.database.autoload_modules import autoload_modules


class TestSubclassModel(TestBaseModel):
    """
    A quick subclass module
    """
    pass


class TestFinalModel(pymodm.MongoModel):
    """
    A final model
    """
    class Meta:
        final = True


class TestAutoloadModulesDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model and removing the image file
        TestBaseModel._mongometa.collection.drop()

    @mock.patch('arvet.database.autoload_modules.importlib.sys.modules')
    @mock.patch('arvet.database.autoload_modules.importlib.import_module')
    def test_tries_to_load_all_the_modules_that_exist(self, mock_import_modules, mock_sys_modules):
        module_names = []

        model = TestBaseModel()
        model.save()
        module_names.append(model.__module__)

        model = TestSubclassModel()
        model.save()
        module_names.append(model.__module__)

        # Make it seem like no modules are loaded
        mock_sys_modules.__contains__.return_value = False

        autoload_modules(TestBaseModel)

        self.assertTrue(mock_import_modules.called)
        for module_name in module_names:
            self.assertIn(mock.call(module_name), mock_import_modules.call_args_list)

    @mock.patch('arvet.database.autoload_modules.importlib.sys.modules')
    @mock.patch('arvet.database.autoload_modules.importlib.import_module')
    def test_only_looks_for_models_for_the_given_ids(self, mock_import_modules, mock_sys_modules):
        base_model = TestBaseModel()
        base_model.save()

        sub_model = TestSubclassModel()
        sub_model.save()

        # Make it seem like no modules are loaded
        mock_sys_modules.__contains__.return_value = False

        autoload_modules(TestBaseModel, ids=[sub_model.pk])

        self.assertTrue(mock_import_modules.called)
        self.assertEqual(1, len(mock_import_modules.call_args_list))
        self.assertNotIn(mock.call(base_model.__module__), mock_import_modules.call_args_list)
        self.assertIn(mock.call(sub_model.__module__), mock_import_modules.call_args_list)

    @mock.patch('arvet.database.autoload_modules.importlib.sys.modules')
    @mock.patch('arvet.database.autoload_modules.importlib.import_module')
    def test_doesnt_load_any_modules_when_given_no_ids(self, mock_import_modules, mock_sys_modules):
        model = TestBaseModel()
        model.save()
        model = TestSubclassModel()
        model.save()

        # Make it seem like no modules are loaded
        mock_sys_modules.__contains__.return_value = False

        autoload_modules(TestBaseModel, ids=[])

        self.assertFalse(mock_import_modules.called)

    @mock.patch('arvet.database.autoload_modules.importlib.sys.modules')
    @mock.patch('arvet.database.autoload_modules.importlib.import_module')
    def test_doesnt_reload_modules_already_loaded(self, mock_import_modules, mock_sys_modules):
        base_model = TestBaseModel()
        base_model.save()

        sub_model = TestSubclassModel()
        sub_model.save()

        # Make it seem like only one module is loaded
        mock_sys_modules.__contains__.side_effect = lambda m: m == base_model.__module__

        autoload_modules(TestBaseModel)

        self.assertTrue(mock_import_modules.called)
        self.assertEqual(1, len(mock_import_modules.call_args_list))
        self.assertNotIn(mock.call(base_model.__module__), mock_import_modules.call_args_list)
        self.assertIn(mock.call(sub_model.__module__), mock_import_modules.call_args_list)

    @mock.patch('arvet.database.autoload_modules.importlib.sys.modules')
    @mock.patch('arvet.database.autoload_modules.importlib.import_module')
    def test_doesnt_load_modules_for_final_models(self, mock_import_modules, mock_sys_modules):
        model = TestFinalModel()
        model.save()

        # Make it seem like no modules are loaded
        mock_sys_modules.__contains__.return_value = False

        autoload_modules(TestFinalModel)
        self.assertFalse(mock_import_modules.called)
