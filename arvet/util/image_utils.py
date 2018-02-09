# Copyright (c) 2017, John Skinner
import os
import typing
import enum
import numpy as np
import PIL.Image


class Interpolation(enum.Enum):
    """
    Enum specifying different interpolation types, for image resizing
    """
    NEAREST = 0
    BOX = 1
    BILINEAR = 2
    BICUBIC = 3


def read_colour(filename: str) -> typing.Union[None, np.ndarray]:
    """
    Load an image from file, however we can.
    :param filename:
    :return:
    """
    if os.path.isfile(filename):
        try:
            pil_image = PIL.Image.open(filename)
        except IOError:
            pil_image = None
        return np.array(pil_image) if pil_image is not None else None
    return None


def read_depth(filename: str) -> typing.Union[None, np.ndarray]:
    """
    Load a depth image from a file.
    Depth images are single channel with higher bit depth (usually 16-bit)
    :param filename:
    :return:
    """
    return read_colour(filename)


def convert_to_grey(image_data: np.ndarray) -> np.ndarray:
    """
    Convert an image to greyscale
    :param image_data: The three channel image data, in RGB format
    :return: The image in greyscale, single channel
    """
    if len(image_data.shape) > 2 and image_data.shape[2] == 3:
        pil_image = PIL.Image.fromarray(to_uint_image(image_data))
        return np.array(pil_image.convert('L'))
    return image_data


def get_bounding_box(image_data: np.ndarray) -> typing.Union[typing.Tuple[int, int, int, int], None]:
    """
    Get the bounding box of the non-zero values in a particular image.
    Call this on a logical image or mask.
    :param image_data: The image data, as a numpy array.
    :return: A tuple of (left, upper, right, lower) pixel coordinates, or None if the image is empty
    """
    if image_data.dtype != np.bool:
        image_data = image_data > 0
    pil_image = PIL.Image.fromarray(np.asarray(image_data, dtype=np.uint8), mode='L')
    return pil_image.getbbox()


def resize(image_data: np.ndarray, new_size: typing.Tuple[int, int],
           interpolation: Interpolation = Interpolation.NEAREST) -> np.ndarray:
    """
    Resize an image to a new resolution

    http://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.resize

    :param image_data: The image to resize
    :param new_size: The new size to change it to.
    :param interpolation: The interpolation mode, as an enum. Default NEAREST.
    :return: The resized image.
    """
    dtype = image_data.dtype

    # Guess the PIL mode best to handle the image
    mode = None
    if dtype == np.uint8:
        if len(image_data.shape) > 2:
            if image_data.shape[2] == 3:
                mode = 'RGB'
            else:
                mode = 'RGBA'
        else:
            mode = 'L'
    if dtype == np.float or dtype == np.float16 or dtype == np.float32 or dtype == np.float64:
        image_data = np.asarray(image_data, dtype=np.float32)
        mode = 'F'
    elif dtype == np.int or dtype == np.int8 or dtype == np.int16 or dtype == np.int32 or dtype == np.int64:
        image_data = np.asarray(image_data, dtype=np.int32)
        mode = 'I'

    pil_image = PIL.Image.fromarray(image_data, mode=mode)

    # Work out which PIL interpolation to use
    pil_interpolation = PIL.Image.NEAREST
    if interpolation == Interpolation.BOX:
        pil_interpolation = PIL.Image.BOX
    elif interpolation == Interpolation.BILINEAR:
        pil_interpolation = PIL.Image.BILINEAR
    elif interpolation == Interpolation.BICUBIC:
        pil_interpolation = PIL.Image.BICUBIC

    # Return, converting back to the original data type
    return np.array(pil_image.resize(new_size, resample=pil_interpolation), dtype=dtype)


def show_image(image_data: np.ndarray, window_name: str = 'temp') -> None:
    """
    Display a particular image to the screen.
    Useful for visualization of results. Note that this cannot handle video or successive frames.
    :param image_data: The image data to display, as an ndarray
    :param window_name: The name of the window, default 'temp'
    :return:
    """
    # PIL basically only handles integer images, so convert to that.
    pil_image = PIL.Image.fromarray(to_uint_image(image_data))
    pil_image.show(window_name)


def to_uint_image(image_data: np.ndarray) -> np.ndarray:
    """
    Convert an image object to an array of uint8 values.
    Many systems and libraries expect this format, so we coerce all other image types
    to this format
    :param image_data: An image, color or grey, in some format (often float range 0-1)
    :return: A uint8 image, range 0 to 255
    """
    if image_data.dtype == np.uint8:
        return image_data
    if (image_data.dtype == np.float or image_data.dtype == np.float16 or
            image_data.dtype == np.float32 or image_data.dtype == np.float64 or
            image_data.dtype == np.double or image_data.dtype == np.bool) and image_data.max() <= 1:
        return (255 * image_data).astype(dtype=np.uint8)
    return image_data.astype(np.uint8)
