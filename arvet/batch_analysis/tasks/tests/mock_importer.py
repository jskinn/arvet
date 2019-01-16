# Copyright (c) 2019, John Skinner
"""
A simple module that meets the requirements for import_dataset_task
Used in test_import_dataset_task
"""
called = False
called_path = None
called_kwargs = None

raise_exception = False
return_value = None


def import_dataset(path, **kwargs):
    global called, called_path, called_kwargs
    global raise_exception, return_value
    called = True
    called_path = path
    called_kwargs = kwargs
    if raise_exception:
        raise ValueError("You told me to")
    return return_value


def reset():
    global called, called_path, called_kwargs
    global raise_exception, return_value
    called = False
    called_path = None
    called_kwargs = None
    raise_exception = False
    return_value = None
