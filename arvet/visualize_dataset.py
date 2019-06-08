#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import argparse
import bson
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import functools

from arvet.config.global_configuration import load_global_config
import arvet.database.connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.core.image_collection import ImageCollection


def make_display_image(image, stereo=False):
    if stereo:
        shape = list(image.data.shape)
        shape[1] *= 2   # Double the width, so that we can
        composite_image = np.zeros(shape, dtype=np.uint8)
        composite_image[:, 0:image.data.shape[1]] = image.data
        if hasattr(image, 'right_data') and image.right_data is not None:
            composite_image[:, image.data.shape[1]:shape[1]] = image.right_data
        elif image.depth_data is not None:
            uint_depth = np.asarray(np.floor(255 * image.depth_data / 4), dtype=np.uint8)
            if len(shape) >= 3:
                for i in range(shape[2]):
                    composite_image[:, image.data.shape[1]:shape[1], i] = uint_depth
            else:
                composite_image[:, image.data.shape[1]:shape[1]] = uint_depth
        return composite_image
    else:
        return image.data


def update_figure(idx, image_source, matplotlib_im, stereo=False, *_, **__):
    _, image = image_source[idx]
    matplotlib_im.set_array(make_display_image(image, stereo))
    return matplotlib_im,  # Comma is important


def visualize_dataset(dataset_id: bson.ObjectId):
    image_collection = ImageCollection.objects.get({'_id': dataset_id})

    fig = pyplot.figure()
    _, first_image = image_collection[0]
    im = pyplot.imshow(make_display_image(first_image,
                                          image_collection.is_stereo_available or image_collection.is_depth_available))
    # Note, we apparently have to store the animation, at least temporarily, or it doesn't play
    ani = animation.FuncAnimation(
        fig, functools.partial(update_figure, image_source=image_collection, matplotlib_im=im,
                               stereo=image_collection.is_stereo_available or image_collection.is_depth_available),
        frames=image_collection.timestamps, blit=True)
    pyplot.show()


def main():
    """
    Visualize a random generated dataset, to make sure we're generating them right
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Visualize a dataset, specified by ID on the command line.')
    parser.add_argument('dataset_id', help='The ID of the dataset to visualize.')
    args = parser.parse_args()

    # Load the configuration
    config = load_global_config('config.yml')

    # Configure the database and the image manager
    dbconn.configure(config['database'])
    im_manager.configure(config['image_manager'])

    visualize_dataset(bson.ObjectId(args.dataset_id))


if __name__ == '__main__':
    main()
