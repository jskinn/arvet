#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import argparse
import bson
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
import functools
import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.core.image


def update_figure(idx, image_source, display_image, *_, **__):
    with image_source:
        image = image_source.get(idx)
    display_image.set_array(image.data)
    return display_image,


def visualize_dataset(db_client, dataset_id):
    s_image_source = db_client.image_source_collection.find_one({'_id': dataset_id})
    if s_image_source is not None:
        image_source = db_client.deserialize_entity(s_image_source)
        if image_source is not None:
            fig = pyplot.figure()
            im = pyplot.imshow(np.zeros((480, 640)))
            # Note, we apparently have to store the animation, at least temporarily, or it doesn't play
            ani = animation.FuncAnimation(
                fig, functools.partial(update_figure, image_source=image_source, display_image=im),
                frames=image_source.timestamps, blit=True)
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

    config = global_conf.load_global_config('config.yml')
    db_client = arvet.database.client.DatabaseClient(config=config)

    visualize_dataset(db_client, bson.ObjectId(args.dataset_id))


if __name__ == '__main__':
    main()
