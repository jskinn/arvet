# Copyright (c) 2017, John Skinner
import enum


class ImageSequenceType(enum.Enum):
    """
    An enum for distinguishing between types of image sequences.
    Some sequences are continuous, like a video sequence,
    while others are just a set of random images.
    Still others are interactive, requiring input to determine the output image
    """
    NON_SEQUENTIAL = 0
    SEQUENTIAL = 1
    INTERACTIVE = 2
