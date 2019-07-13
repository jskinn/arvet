import typing
import sys
import bson
import importlib
import logging
from pymodm.base.models import MongoModelMetaclass


def autoload_modules(model: MongoModelMetaclass, ids: typing.List[bson.ObjectId] = None):
    """

    :param model:
    :param ids:
    :return:
    """
    if model._mongometa.final:
        # The model is final, and has no subclasses, don't load anything
        return

    query = {'_cls': {'$exists': True}}
    if ids is not None:
        if len(ids) == 1:
            query['_id'] = ids[0]
        else:
            query['_id'] = {'$in': ids}
    # Use the mongo collection directly to find the set of all the classes of models in the collection
    cursor = model._mongometa.collection.find(query, {'_id': False, '_cls': True})
    for result in cursor:
        # For each class we found, try and load the module it contains
        type_name = result['_cls']
        module_name = type_name.rpartition('.')[0]
        if module_name is not '' and module_name not in sys.modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                logging.getLogger(__name__).warning(
                    "Could not import module {0} containing model {1},"
                    "models of that type will not be returned".format(module_name, type_name))
                pass
