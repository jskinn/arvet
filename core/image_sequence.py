import enum


class ImageSequenceType(enum.Enum):
    """
    An enum for distinguishing between types of image sequences.
    Some sequences are continuous, like a video sequence,
    while others are just a set of random images.
    """
    NON_SEQUENTIAL = 0
    SEQUENTIAL = 1
