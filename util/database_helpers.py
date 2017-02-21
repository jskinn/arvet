

def query_to_dot_notation(query):
    """
    Recursively transform a query containing nested dicts to mongodb dot notation.
    That is,
    {'test': {'a': 1, 'b': 2}} becomes {'test.a': 1, 'test.b': 2}

    Note that this modifies the parameter, and returns it.
    you can either call this as query_to_dot_notation(query) or query = query_to_dot_notation({})

    :param query: The base query as a dict
    :return: query.
    """
    initial_keys = list(query.keys())
    for key in initial_keys:
        if isinstance(query[key], dict):
            query_to_dot_notation(query[key])
            for inner_key, inner_value in query[key].items():
                query[key+'.'+inner_key] = inner_value
            del query[key]
    return query