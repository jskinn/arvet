#Copyright (c) 2017, John Skinner
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

    def test_flatten_array(self):
        result = dh.query_to_dot_notation({
            'a': [{
                'b': 1.12
            },{
                'a': 1.21,
                'b': 1.22
            },{
                'c': 1.33
            }]
        }, flatten_arrays=True)
        self.assertEqual(result, {'a.0.b': 1.12, 'a.1.a': 1.21, 'a.1.b': 1.22, 'a.2.c': 1.33})

    def test_flatten_array_recursive(self):
        result = dh.query_to_dot_notation({
            'a': [{
                'b': 1.12,
                'd': [{
                    'a': 1.1411,
                    'b': 1.1412,
                },{
                    'b': 1.1422
                },{
                    'a': 1.1431,
                    'b': 1.1432
                },{
                    'c': 1.1443
                }]
            },{
                'a': 1.21,
                'd': 1.22
            }]
        }, flatten_arrays=True)
        self.assertEqual(result, {
            'a.0.b': 1.12,
            'a.0.d.0.a': 1.1411,
            'a.0.d.0.b': 1.1412,
            'a.0.d.1.b': 1.1422,
            'a.0.d.2.a': 1.1431,
            'a.0.d.2.b': 1.1432,
            'a.0.d.3.c': 1.1443,
            'a.1.a': 1.21,
            'a.1.d': 1.22
        })

    def test_flatten_array_does_not_flatten_arrays_that_are_not_arrays_of_dicts(self):
        result = dh.query_to_dot_notation({
            'a': (11, 12, 13),
            'b': [{
                'b': 1.12,
                'd': [{
                    'a': 1.1411,
                    'b': (1.14121, 1.14122)
                },{
                    'b': 1.1422
                },{
                    'a': (1.14311,),
                    'b': 1.1432
                },{
                    'c': 1.1443
                }]
            },{
                'a': 1.21,
                'd': 1.22
            }]
        }, flatten_arrays=True)
        self.assertEqual({
            'a.0': 11,
            'a.1': 12,
            'a.2': 13,
            'b.0.b': 1.12,
            'b.0.d.0.a': 1.1411,
            'b.0.d.0.b.0': 1.14121,
            'b.0.d.0.b.1': 1.14122,
            'b.0.d.1.b': 1.1422,
            'b.0.d.2.a.0': 1.14311,
            'b.0.d.2.b': 1.1432,
            'b.0.d.3.c': 1.1443,
            'b.1.a': 1.21,
            'b.1.d': 1.22
        }, result)
