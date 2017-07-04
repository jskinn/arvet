

def query_to_dot_notation(query, flatten_arrays=False):
    """
    Recursively transform a query containing nested dicts to mongodb dot notation.
    That is,
    {'test': {'a': 1, 'b': 2}} becomes {'test.a': 1, 'test.b': 2}

    Note that this modifies the parameter, and returns it.
    you can either call this as query_to_dot_notation(query) or query = query_to_dot_notation({})

    :param query: The base query as a dict
    :param flatten_arrays: Should arrays be flattened as well. Defaults to false
    :return: query.
    """
    initial_keys = list(query.keys())
    for key in initial_keys:
        if isinstance(query[key], dict):
            query_to_dot_notation(query[key], flatten_arrays=flatten_arrays)
            for inner_key, inner_value in query[key].items():
                query[key+'.'+inner_key] = inner_value
            del query[key]
        elif flatten_arrays and (isinstance(query[key], list) or isinstance(query[key], tuple)):
            for idx, elem in enumerate(query[key]):
                if isinstance(elem, dict):
                    query_to_dot_notation(elem, flatten_arrays=flatten_arrays)
                    for inner_key, inner_value in elem.items():
                        query[key+'.'+str(idx)+'.'+inner_key] = inner_value
            del query[key]
    return query