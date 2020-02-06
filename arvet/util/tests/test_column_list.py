import unittest
import unittest.mock as mock
import numpy as np
from types import GeneratorType
from operator import attrgetter
from arvet.util.column_list import ColumnList


class TestColumnList(unittest.TestCase):

    def test_contains_kwargs(self):
        columns = ColumnList(
            foo=attrgetter('foo'),
            bar=attrgetter('bar')
        )
        self.assertIn('foo', columns)
        self.assertIn('bar', columns)
        self.assertNotIn('baz', columns)

    def test_contains_parent_kwargs(self):
        parent1 = ColumnList(
            parent1col=attrgetter('foo')
        )
        parent2 = ColumnList(
            parent2col=attrgetter('foo')
        )
        columns = ColumnList(
            parent1, parent2,
            foo=attrgetter('foo'),
            bar=attrgetter('bar')
        )
        self.assertIn('foo', columns)
        self.assertIn('bar', columns)
        self.assertIn('parent1col', columns)
        self.assertIn('parent2col', columns)
        self.assertNotIn('baz', columns)

    def test_keys_returns_generator(self):
        columns = ColumnList(
            foo=attrgetter('foo'),
            bar=attrgetter('bar')
        )
        self.assertIsInstance(columns.keys(), GeneratorType)

    def test_keys_generator_kwargs(self):
        columns = ColumnList(
            foo=attrgetter('foo'),
            bar=attrgetter('bar')
        )
        keys = set(columns.keys())
        self.assertEqual({'foo', 'bar'}, keys)

    def test_keys_returns_parent_kwargs(self):
        parent1 = ColumnList(
            parent1col=attrgetter('foo')
        )
        parent2 = ColumnList(
            parent2col=attrgetter('foo')
        )
        columns = ColumnList(
            parent1, parent2,
            foo=attrgetter('foo'),
            bar=attrgetter('bar')
        )
        keys = set(columns.keys())
        self.assertEqual({'parent1col', 'parent2col', 'foo', 'bar'}, keys)

    def test_get_value_runs_getter_on_obj(self):
        expected_result = 'never gonna give you up'
        getter_func = mock.Mock()
        getter_func.return_value = expected_result
        obj = object()

        columns = ColumnList(foo=getter_func)
        result = columns.get_value(obj, 'foo')
        self.assertTrue(getter_func.called)
        self.assertEqual(mock.call(obj), getter_func.call_args)
        self.assertEqual(expected_result, result)

    def test_get_value_returns_NaN_for_none_columns(self):
        obj = object()

        columns = ColumnList(foo=None, bar=None)
        result = columns.get_value(obj, 'foo')
        self.assertTrue(np.isnan(result))

    def test_get_value_raises_keyerror_if_not_valid_column(self):
        columns = ColumnList(
            foo=attrgetter('foo'),
        )
        with self.assertRaises(KeyError):
            columns.get_value(object(), 'bar')

    def test_get_value_returns_parent_value(self):
        expected_result = 'never gonna give you up'
        getter_func = mock.Mock()
        getter_func.return_value = expected_result
        obj = object()

        parent = ColumnList(bar=getter_func)
        columns = ColumnList(parent, foo=attrgetter('foo'))
        result = columns.get_value(obj, 'bar')
        self.assertTrue(getter_func.called)
        self.assertEqual(mock.call(obj), getter_func.call_args)
        self.assertEqual(expected_result, result)

    def test_get_value_prefers_child_columns(self):
        expected_result = 'never gonna give you up'
        parent_getter_func = mock.Mock()
        getter_func = mock.Mock()
        getter_func.return_value = expected_result
        obj = object()

        parent = ColumnList(foo=parent_getter_func)
        columns = ColumnList(parent, foo=getter_func)
        result = columns.get_value(obj, 'foo')
        self.assertFalse(parent_getter_func.called)
        self.assertTrue(getter_func.called)
        self.assertEqual(mock.call(obj), getter_func.call_args)
        self.assertEqual(expected_result, result)
