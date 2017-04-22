import unittest
import util.database_helpers as dh


class TestDatabaseHelpers(unittest.TestCase):

    def test_combines_keys(self):
        result = dh.query_to_dot_notation({'a': {'b': 1}})
        self.assertEqual(result, {'a.b': 1})

    def test_passes_through_non_nested_keys(self):
        result = dh.query_to_dot_notation({'a': 1})
        self.assertEqual(result, {'a': 1})

    def test_handles_nested_and_non_nested_keys(self):
        result = dh.query_to_dot_notation({'a': 1, 'b': { 'a': 2.1 }})
        self.assertEqual(result, {'a': 1, 'b.a': 2.1})

    def test_works_recursively(self):
        result = dh.query_to_dot_notation({'a': {'b': {'c': {'d': 1}}}})
        self.assertEqual(result, {'a.b.c.d': 1})

    def test_big(self):
        result = dh.query_to_dot_notation({
            'a': 1,
            'b': 2,
            'c': {
                'a': 3.1,
                'b': {
                    'a': 3.21,
                    'b': 3.22
                },
                'c': {
                    'a': 3.31,
                    'b': 3.32,
                    'c': 3.33
                },
                'd': {
                    'a': 3.41,
                    'b': {
                        'a': 3.421
                    }
                }
            },
            'd': {
                'a': {
                    'a': {
                        'a': { # 4.111
                            'a': {
                                'a': 4.11111
                            }
                        },
                        'b': {  # 4.112
                            'a': {
                                'a': {  # 4.11211
                                    'a': {
                                        'a': 4.1121111
                                    }
                                },
                                'b': 4.11212
                            }
                        }
                    }
                }
            },
            'e': 5,
            'f': {
                'a': 6.1,
                'b': {
                    'a': 6.21
                }
            }
        })
        self.assertEqual(result, {
            'a': 1,
            'b': 2,
            'c.a': 3.1,
            'c.b.a': 3.21,
            'c.b.b': 3.22,
            'c.c.a': 3.31,
            'c.c.b': 3.32,
            'c.c.c': 3.33,
            'c.d.a': 3.41,
            'c.d.b.a': 3.421,
            'd.a.a.a.a.a': 4.11111,
            'd.a.a.b.a.a.a.a': 4.1121111,
            'd.a.a.b.a.b': 4.11212,
            'e': 5,
            'f.a': 6.1,
            'f.b.a': 6.21
        })
