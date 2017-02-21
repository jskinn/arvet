import collections


def defaults(base, *args, modify_base=True):
    """
    Utility for merging dicts or dict-like objects.
    The first positional parameter is the base object,
    and further positional arguments will be merged into the base object as default values.
    This will not override existing keys in the base object.
    Priority order for merging is the order they are specified as positional arguments.

    Example:
    defaults({'a': 1}, {'a': 1.5, 'b': 2}, {'b': 2.5, 'c': 3})
    returns {'a': 1, 'b': 2, 'c': 3}

    By default, this modifies the base object, specify modify_base=False keyword to clone the object
    :param base: dict
    :param args: list of dict
    :param modify_base: bool
    :return: the merged dict
    :rtype: dict
    """
    if modify_base:
        result = base
    else:
        result = dict(base)

    for arg in args:
        for key, value in arg.items():
            if not key in result:
                result[key] = value
            elif isinstance(result[key], dict):
                result[key] = defaults(result[key], value)
    return result


def split_period_keys(base):
    result = {}
    for key, value in base.items():
        subkeys = key.split('.')
        inner = result
        for idx in range(0, len(subkeys)):
            if idx == len(subkeys) - 1:
               inner[subkeys[idx]] = value
            else:
                if subkeys[idx] not in inner:
                    inner[subkeys[idx]] = {}
                elif not isinstance(inner[subkeys[idx]], collections.Mapping):
                    break   # we've hit a non-dict before we ran out of keys, break
                inner = inner[subkeys[idx]]
    return result
