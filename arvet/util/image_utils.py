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
        pil_image = PIL.Image.fromarray(image_data)
        return np.array(pil_image.convert('L'))
    return image_data


def get_bounding_box(image_data: np.ndarray) -> typing.Union[typing.Tuple[int, int, int, int], None]:
    """
    Get the bounding box of the non-zero values in a particular image.
    Call this on a logical image or mask.
    :param image_data: The image data, as a numpy array.
    :return: A tuple of (left, upper, right, lower) pixel coordinates, or None if the image is empty
    """
    if image_data.dtype == np.bool:
        # PIL won't handle logical np arrays, it expects everything as uint8
        image_data = image_data.astype(dtype=np.uint8)
    pil_image = PIL.Image.fromarray(image_data)
    return pil_image.getbbox()


def resize_image(image_data: np.ndarray, new_size: typing.Tuple[int, int],
                 interpolation: Interpolation = Interpolation.NEAREST) -> np.ndarray:
    """
    Resize an image to a new resolution

    http://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.resize

    :param image_data: The image to resize
    :param new_size: The new size to change it to.
    :param interpolation: The interpolation mode, as an enum. Default NEAREST.
    :return: The resized image.
    """
    pil_image = PIL.Image.fromarray(image_data)
    pil_interpolation = PIL.Image.NEAREST
    if interpolation == Interpolation.BOX:
        pil_interpolation = PIL.Image.BOX
    elif interpolation == Interpolation.BILINEAR:
        pil_interpolation = PIL.Image.BILINEAR
    elif interpolation == Interpolation.BICUBIC:
        pil_interpolation = PIL.Image.BICUBIC
    return np.array(pil_image.resize(new_size, resample=pil_interpolation))


def show_image(image_data: np.ndarray, window_name: str = 'temp') -> None:
    """
    Display a particular image to the screen.
    Useful for visualization of results. Note that this cannot handle video or successive frames.
    :param image_data: The image data to display, as an ndarray
    :param window_name: The name of the window, default 'temp'
    :return:
    """
    # PIL basically only handles integer images, so convert to that.
    if (image_data.dtype == np.float or image_data.dtype == np.float16 or
            image_data.dtype == np.float32 or image_data.dtype == np.float64 or
            image_data.dtype == np.double) and image_data.max() <= 1:
        image_data = (255 * image_data).astype(dtype=np.uint8)
    elif image_data.dtype == np.bool:
        image_data = image_data.astype(dtype=np.uint8)
    pil_image = PIL.Image.fromarray(image_data)
    pil_image.show(window_name)
