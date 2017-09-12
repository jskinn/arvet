#Copyright (c) 2017, John Skinner
import unittest
import util.dict_utils as du


class TestDictUtils(unittest.TestCase):

    def test_defaults_noop(self):
        result = du.defaults({'a': 1})
        self.assertEqual(result, {'a': 1})

    def test_defaults_example(self):
        # This is the example in the doc
        result = du.defaults({'a': 1}, {'a': 1.5, 'b': 2}, {'b': 2.5, 'c': 3})
        self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3})

    def test_defaults_order(self):
        result = du.defaults({'a': 1}, {'b': 2.5, 'c': 3}, {'a': 1.5, 'b': 2})
        self.assertEqual(result, {'a': 1, 'b': 2.5, 'c': 3})

    def test_defaults_changes_base(self):
        base = {'a': 1}
        result = du.defaults(base, {'a': 1.5, 'b': 2}, {'b': 2.5, 'c': 3})
        self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(base, result)

    def test_defaults_suppress_base_change(self):
        base = {'a': 1}
        result = du.defaults(base, {'a': 1.5, 'b': 2}, {'b': 2.5, 'c': 3}, modify_base=False)
        self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(base, {'a': 1})

    def test_defaults_list(self):
        result = du.defaults({'a': 1, 'b':2}, {'a': 1.1, 'c': 3.1}, {'a': 1.2, 'd': 4.2}, {'a': 1.3, 'e': 5.3})
        self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3.1, 'd': 4.2, 'e': 5.3})

    def test_works_recursively(self):
        result = du.defaults({
            'a': {
                'a.1': 1,
                'a.2': 2,
                'a.a': {
                    'a.a.1': 1,
                    'a.a.2': 2
                }
            },
            'b': {
                'b.1': 1,
                'b.2': 2
            }
        }, {
            'a': {
                'a.1': 1.1,
                'a.3': 3.1,
                'a.a': {
                    'a.a.1': 1.1,
                    'a.a.3': 3.1
                }
            },
            'c': {
                'c.1': 1.1,
                'c.2': 2.1
            }
        })
        self.assertEqual(result, {
            'a': {
                'a.1': 1,
                'a.2': 2,
                'a.3': 3.1,
                'a.a': {
                    'a.a.1': 1,
                    'a.a.2': 2,
                    'a.a.3': 3.1
                }
            },
            'b': {
                'b.1': 1,
                'b.2': 2
            },
            'c': {
                'c.1': 1.1,
                'c.2': 2.1
            }
        })

    def test_split_period_keys(self):
        self.assertEqual(du.split_period_keys({'a.b': 1}), {'a': {'b': 1}})

    def test_split_period_keys_noop(self):
        self.assertEqual(du.split_period_keys({'a': 1}), {'a': 1})

    def test_split_period_keys_merges(self):
        self.assertEqual(
            du.split_period_keys({'a.b': 1, 'a.c': 2, 'a.d': 3}),
            {'a': {'b': 1, 'c': 2, 'd': 3} })

    def test_split_period_handles_multiple(self):
        self.assertEqual(
            du.split_period_keys({
                'a.a.a': 1,
                'a.a.b': 2,
                'a.a.c': 3,
                'a.b.a': 4,
                'a.b.b': 5,
                'a.b.c': 6,
                'a.c.a': 7,
                'a.c.b': 8,
                'a.c.c': 9,

                'b.a.a': 10,
                'b.a.b': 11,
                'b.b.a': 12,
                'b.b.b': 13,
                'b.b.c': 14,
            }),{''
                'a': {
                    'a': {'a': 1, 'b': 2, 'c':3},
                    'b': {'a': 4, 'b': 5, 'c':6},
                    'c': {'a': 7, 'b': 8, 'c':9}
                },
                'b': {
                    'a': {'a': 10, 'b': 11},
                    'b': {'a': 12, 'b': 13, 'c':14}
                }
            })

    def test_split_period_keys_doesnt_override(self):
        self.assertEqual(du.split_period_keys({'a.b': 1, 'a': 2}), {'a': 2})

    def test_split_period_keys_copies(self):
        base = {'a.a': 1, 'a.b': 2, 'a.c': 3, 'b.a': 4, 'b.b': 5, 'b.c': 6}
        result = du.split_period_keys(base)
        self.assertEqual(result, {
            'a': {'a': 1, 'b': 2, 'c': 3},
            'b': {'a': 4, 'b': 5, 'c': 6}
        })
        self.assertEqual(base, {'a.a': 1, 'a.b': 2, 'a.c': 3, 'b.a': 4, 'b.b': 5, 'b.c': 6})
