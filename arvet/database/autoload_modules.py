import typing
import sys
import importlib
import logging
from pymodm import MongoModel


def autoload_modules(model: typing.Type[MongoModel], ids: typing.List[typing.Any] = None) -> None:
    """
    Find and automatically load the subtypes of a given model class.
    Used to allow me to use base model types with many different subclasses,
    and let the experiments hold custom code.
    Systems, Metrics, and Experiments are some examples of this pattern.

    :param model: The base model to query
    :param ids: An optional list of ids, will limit the loaded models to those with the given ids
    :return: None
    """
    # First, get the pymodm metainformation
    if hasattr(model, '_mongometa'):
        mongometa = model._mongometa
    else:
        raise RuntimeError("Object is a pymodm model, but doesn't have meta information?")

    # If the model is final, and has no subclasses, don't load anything
    if mongometa.final:
        return

    # Work out which classes we want to load
    query = {'_cls': {'$exists': True}}
    if ids is not None:
        if len(ids) == 1:
            query['_id'] = ids[0]
        else:
            query['_id'] = {'$in': ids}

    # Use the mongo collection directly to find the set of all the classes of models in the collection
    cursor = mongometa.collection.distinct('_cls', query)
    for type_name in cursor:
        # For each class we found, try and load the module it contains
        module_name = type_name.rpartition('.')[0]
        if module_name is not '' and module_name not in sys.modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                logging.getLogger(__name__).warning(
                    "Could not import module {0} containing model {1},"
                    "models of that type will not be returned".format(module_name, type_name))
                pass


ModelType = typing.TypeVar('ModelType', bound=MongoModel)


def get_model_classes(model: typing.Type[ModelType], ids: typing.Collection[typing.Any]) \
        -> typing.List[typing.Type[ModelType]]:
    """
    Get the model class objects for specific objects, useful to run class methods
    Will try and load the models if they are missing.
    Missing classes or modules are logged, but do not raise errors.

    :param model: The base model type, providing the meta-information like which collection to use
    :param ids: The list of ids to find the classes of
    :return: A list of model objects
    """
    # First, get the pymodm metainformation
    if hasattr(model, '_mongometa'):
        mongometa = model._mongometa
    else:
        raise RuntimeError("Object is a pymodm model, but doesn't have meta information?")

    # If the model is final, it has no subclasses, so there is nothing to find.
    if mongometa.final:
        # The model is final, and has no subclasses, return the model
        return [model]

    # Work out which classes we want to load
    query = {'_cls': {'$exists': True}}
    if ids is not None:
        if len(ids) == 1:
            query['_id'] = next(iter(ids))  # Is this the cleanest way to get the single element?
        else:
            query['_id'] = {'$in': list(ids)}

    # Use the mongo collection directly to find the set of all the classes of models in the collection
    model_classes = []
    cursor = mongometa.collection.distinct('_cls', query)
    for type_name in cursor:
        # For each class we found, try and load the module it contains
        module_name, _, class_name = type_name.rpartition('.')

        # Step 1: find the module containing the class
        if module_name is '':
            continue
        else:
            if module_name in sys.modules:
                # Module is already loaded, read it
                module = sys.modules[module_name]
            else:
                # Module is not loaded, try and load it
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    logging.getLogger(__name__).warning(
                        "Could not import module {0} containing model {1},"
                        "models of that type will not be returned".format(module_name, type_name))
                    continue

        # Then, if we found a found a module, try and get the model from it
        if module is not None:
            if hasattr(module, class_name):
                model_classes.append(getattr(module, class_name))
            else:
                logging.getLogger(__name__).warning(
                    "Could not find model class {0} in module {1}".format(class_name, module_name))
    return model_classes
