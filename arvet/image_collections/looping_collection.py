# Copyright (c) 2017, John Skinner
import pymodm.fields as fields
from arvet.core.image_collection import ImageCollection


class LoopingCollection(ImageCollection):
    """
    A wrapper around an image collection, which makes it repeat.
    Very simple.
    """
    repeats = fields.IntegerField()

    def __len__(self):
        return self.repeats * super().__len__()

    def __getitem__(self, item):
        return

    def __iter__(self):
        iterator = super().__iter__()
        for _ in range(self.repeats):
            yield from iterator
