#!/usr/bin/env python3
# Copyright (c) 2017, John Skinner
import bson
import arvet.config.global_configuration as global_conf
import arvet.database.client
import arvet.core.image

try:
    import cv2
except ImportError:
    cv2 = None


def visualize_dataset(db_client, dataset_id):
    if cv2 is None:
        return
    s_image_source = db_client.image_source_collection.find_one({'_id': dataset_id})
    if s_image_source is not None:
        image_source = db_client.deserialize_entity(s_image_source)
        with image_source:
            while not image_source.is_complete():
                image, _ = image_source.get_next_image()
                cv2.imshow('rgb', image.data[:, :, ::-1])
                if isinstance(image, arvet.core.image.StereoImage):
                    cv2.imshow('right', image.right_data[:, :, ::-1])
                cv2.waitKey(100)


def visualize_generated_dataset(db_client):
    if cv2 is None:
        return
    generate_tasks = db_client.tasks_collection.find({
        '_type': 'arvet.batch_analysis.tasks.generate_dataset_task.GenerateDatasetTask', 'state': 2}).limit(2)
    for s_generate_task in generate_tasks:
        task = db_client.deserialize_entity(s_generate_task)
        result_ids = task.result
        if result_ids is None:
            continue
        elif isinstance(result_ids, bson.ObjectId):
            result_ids = [result_ids]
        for result_id in result_ids:
            visualize_dataset(db_client, result_id)


def main():
    """
    Visualize a random generated dataset, to make sure we're generating them right
    :return:
    """
    config = global_conf.load_global_config('config.yml')
    db_client = arvet.database.client.DatabaseClient(config=config)

    visualize_dataset(db_client, bson.ObjectId("5a00854936ed1e1fa9a4ae19"))


if __name__ == '__main__':
    main()
