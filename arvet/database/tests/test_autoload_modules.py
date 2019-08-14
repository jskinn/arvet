import unittest
import unittest.mock as mock
import sys
import importlib
import pymodm
import arvet.database.tests.database_connection as dbconn
from .mock_base_model import TestBaseModel
from arvet.database.autoload_modules import autoload_modules, get_model_classes


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
    _orig_sys_modules = None
    _orig_import_module = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls._orig_sys_modules = sys.modules
        cls._orig_import_module = importlib.import_module

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model and removing the image file
        TestBaseModel._mongometa.collection.drop()

        # check we haven't contaminated global state
        assert sys.modules == cls._orig_sys_modules
        assert importlib.import_module == cls._orig_import_module

    @mock.patch('arvet.database.autoload_modules.sys', spec=['modues'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_tries_to_load_all_the_modules_that_exist(self, mock_import_module, mock_sys):
        module_names = []

        model = TestBaseModel()
        model.save()
        module_names.append(model.__module__)

        model = TestSubclassModel()
        model.save()
        module_names.append(model.__module__)

        # Make it seem like no modules are loaded
        mock_sys.modules = {}

        autoload_modules(TestBaseModel)

        self.assertTrue(mock_import_module.called)
        for module_name in module_names:
            self.assertIn(mock.call(module_name), mock_import_module.call_args_list)

    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_only_looks_for_models_for_the_given_ids(self, mock_import_module, mock_sys):
        base_model = TestBaseModel()
        base_model.save()

        sub_model = TestSubclassModel()
        sub_model.save()

        # Make it seem like no modules are loaded
        mock_sys.modules = {}

        autoload_modules(TestBaseModel, ids=[sub_model.pk])

        self.assertTrue(mock_import_module.called)
        self.assertEqual(1, len(mock_import_module.call_args_list))
        self.assertNotIn(mock.call(base_model.__module__), mock_import_module.call_args_list)
        self.assertIn(mock.call(sub_model.__module__), mock_import_module.call_args_list)

    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_doesnt_load_any_modules_when_given_no_ids(self, mock_import_module, mock_sys):
        model = TestBaseModel()
        model.save()
        model = TestSubclassModel()
        model.save()

        # Make it seem like no modules are loaded
        mock_sys.modules = {}

        autoload_modules(TestBaseModel, ids=[])

        self.assertFalse(mock_import_module.called)

    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_doesnt_reload_modules_already_loaded(self, mock_import_module, mock_sys):
        base_model = TestBaseModel()
        base_model.save()

        sub_model = TestSubclassModel()
        sub_model.save()

        # Make it seem like only one module is loaded
        mock_sys.modules = {base_model.__module__: mock.Mock()}

        autoload_modules(TestBaseModel)

        self.assertTrue(mock_import_module.called)
        self.assertEqual(1, len(mock_import_module.call_args_list))
        self.assertNotIn(mock.call(base_model.__module__), mock_import_module.call_args_list)
        self.assertIn(mock.call(sub_model.__module__), mock_import_module.call_args_list)

    @mock.patch('arvet.database.autoload_modules.sys', spec='modules')
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_doesnt_load_modules_for_final_models(self, mock_import_module, mock_sys):
        model = TestFinalModel()
        model.save()

        # Make it seem like no modules are loaded
        mock_sys.modules = {}

        autoload_modules(TestFinalModel)
        self.assertFalse(mock_import_module.called)


class TestGetModelClassesDatabase(unittest.TestCase):
    _orig_sys_modules = None
    _orig_import_module = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls._orig_sys_modules = sys.modules
        cls._orig_import_module = importlib.import_module

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model and removing the image file
        TestBaseModel._mongometa.collection.drop()

        # check we haven't contaminated global state
        assert sys.modules == cls._orig_sys_modules
        assert importlib.import_module == cls._orig_import_module

    def test_returns_model_classes_for_ids(self):
        model_ids = []

        model = TestBaseModel()
        model.save()
        model_ids.append(model.pk)

        model = TestSubclassModel()
        model.save()
        model_ids.append(model.pk)

        result = get_model_classes(TestBaseModel, model_ids)
        self.assertEqual(2, len(result))
        self.assertIn(TestBaseModel, result)
        self.assertIn(TestSubclassModel, result)

    def test_returns_models_for_only_requested_ids(self):
        model_ids = []

        # put this model in the database, but we're not going to request it
        model = TestBaseModel()
        model.save()

        model = TestSubclassModel()
        model.save()
        model_ids.append(model.pk)

        result = get_model_classes(TestBaseModel, model_ids)
        self.assertEqual(1, len(result))
        self.assertNotIn(TestBaseModel, result)
        self.assertIn(TestSubclassModel, result)

    def test_returns_unique_models(self):
        model_ids = []

        model = TestBaseModel()
        model.save()

        # Make many objects of the same type, we're going to request them all
        for _ in range(3):
            model = TestSubclassModel()
            model.save()
            model_ids.append(model.pk)

        result = get_model_classes(TestBaseModel, model_ids)
        self.assertEqual(1, len(result))
        self.assertNotIn(TestBaseModel, result)
        self.assertIn(TestSubclassModel, result)

    def test_accepts_set_of_ids(self):
        model_ids = []

        model = TestBaseModel()
        model.save()
        model_ids.append(model.pk)

        model = TestSubclassModel()
        model.save()
        model_ids.append(model.pk)

        result = get_model_classes(TestBaseModel, set(model_ids))
        self.assertEqual(2, len(result))
        self.assertIn(TestBaseModel, result)
        self.assertIn(TestSubclassModel, result)

        result = get_model_classes(TestBaseModel, {model_ids[0]})
        self.assertEqual([TestBaseModel], result)

    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_returns_but_doesnt_try_and_load_subclasses_of_final_models(self, mock_import_module, mock_sys):
        model_ids = []
        module_names = []

        model = TestFinalModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        # Make it seem like no modules are loaded, but that we can load modules
        mock_sys.modules = {}
        mock_import_module.return_value = make_mock_module(TestFinalModel)

        # Mock the collection, to make sure we're not calling anything on it
        with mock.patch.object(TestFinalModel, '_mongometa') as mock_meta:
            mock_meta.final = True
            mock_collection = mock.Mock()
            mock_meta.collection = mock_collection

            result = get_model_classes(TestFinalModel, model_ids)
            self.assertEqual(1, len(result))
            self.assertIn(TestFinalModel, result)
            self.assertFalse(mock_import_module.called)
            self.assertEqual([], mock_collection.method_calls)

    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_tries_to_load_models_that_dont_exist(self, mock_import_module, mock_sys):
        model_ids = []
        module_names = []

        model = TestBaseModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        model = TestSubclassModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        # Make it seem like no modules are loaded, but that we can load modules
        mock_sys.modules = {}
        mock_import_module.return_value = make_mock_module(TestBaseModel, TestSubclassModel)

        result = get_model_classes(TestBaseModel, model_ids)
        self.assertEqual(2, len(result))
        self.assertIn(TestBaseModel, result)
        self.assertIn(TestSubclassModel, result)

        self.assertTrue(mock_import_module.called)
        for module_name in module_names:
            self.assertIn(mock.call(module_name), mock_import_module.call_args_list)

    @mock.patch('arvet.database.autoload_modules.logging', spec=['getLogger'])
    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_fails_with_log_on_unimportable_modules(self, mock_import_module, mock_sys, mock_logging):
        model_ids = []
        module_names = []

        model = TestBaseModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        model = TestSubclassModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        # Make it seem like no modules are loaded, but that we can load modules
        mock_logger = make_mock_logger(mock_logging)
        mock_sys.modules = {}
        mock_module = make_mock_module(TestBaseModel, TestSubclassModel)

        def occasional_error(name):
            if name == TestBaseModel.__module__:
                return mock_module
            raise ImportError("Test error")
        mock_import_module.side_effect = occasional_error

        # Check we returned the class that we can import successfully
        result = get_model_classes(TestBaseModel, model_ids)
        self.assertEqual(1, len(result))
        self.assertIn(TestBaseModel, result)
        self.assertNotIn(TestSubclassModel, result)

        # Check we still tried to import all the modules
        self.assertTrue(mock_import_module.called)
        for module_name in module_names:
            self.assertIn(mock.call(module_name), mock_import_module.call_args_list)

        # Check that the fact that we couldn't load a module was logged
        self.assertTrue(mock_logger.warning.called)
        log_msg = mock_logger.warning.call_args[0][0]
        self.assertIn(TestSubclassModel.__module__, log_msg)

    @mock.patch('arvet.database.autoload_modules.logging', spec=['getLogger'])
    @mock.patch('arvet.database.autoload_modules.sys', spec=['modules'])
    @mock.patch('arvet.database.autoload_modules.importlib.import_module', autospec=True)
    def test_fails_with_log_on_missing_classes(self, mock_import_module, mock_sys, mock_logging):
        model_ids = []
        module_names = []

        model = TestBaseModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        model = TestSubclassModel()
        model.save()
        model_ids.append(model.pk)
        module_names.append(model.__module__)

        # Make it seem like no modules are loaded, but that we can load modules
        mock_logger = make_mock_logger(mock_logging)
        mock_sys.modules = {}
        mock_import_module.return_value = make_mock_module(TestBaseModel)

        # Check we returned the class that we can import successfully
        result = get_model_classes(TestBaseModel, model_ids)
        self.assertEqual(1, len(result))
        self.assertIn(TestBaseModel, result)
        self.assertNotIn(TestSubclassModel, result)

        # Check we still tried to import all the modules
        self.assertTrue(mock_import_module.called)
        for module_name in module_names:
            self.assertIn(mock.call(module_name), mock_import_module.call_args_list)

        # Check that the fact that we couldn't load a module was logged
        self.assertTrue(mock_logger.warning.called)
        log_msg = mock_logger.warning.call_args[0][0]
        self.assertIn(TestSubclassModel.__name__, log_msg)


def make_mock_module(*module_classes):
    mock_module = mock.Mock(spec=[module_class.__name__ for module_class in module_classes])
    for module_class in module_classes:
        setattr(mock_module, module_class.__name__, module_class)
    return mock_module


def make_mock_logger(mock_logging):
    mock_logger = mock.Mock(spec=['warning', 'error', 'debug'])
    mock_logging.getLogger.return_value = mock_logger
    return mock_logger
